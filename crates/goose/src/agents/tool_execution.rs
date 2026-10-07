use async_stream::try_stream;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use rmcp::model::CallToolResult;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use std::path::PathBuf;

use crate::config::permission::PermissionLevel;
use crate::conversation::message::{ActionRequiredData, Message, MessageContent};
use crate::mcp_utils::ToolResult;
use crate::permission::Permission;
use rmcp::model::{ContentBlock, ServerNotification};

#[derive(Clone)]
pub(crate) struct ToolCallNotificationEmitter {
    sender: mpsc::Sender<ServerNotification>,
}

impl ToolCallNotificationEmitter {
    pub(crate) fn new(sender: mpsc::Sender<ServerNotification>) -> Self {
        Self { sender }
    }

    pub(crate) fn emit_best_effort(&self, notification: ServerNotification) {
        // Do not let a slow notification consumer delay tool execution.
        let _ = self.sender.try_send(notification);
    }
}

/// Context passed through the tool call dispatch chain.
#[derive(Clone)]
pub struct ToolCallContext {
    pub session_id: String,
    pub working_dir: Option<PathBuf>,
    pub tool_call_request_id: Option<String>,
    notification_emitter: Option<ToolCallNotificationEmitter>,
}

impl ToolCallContext {
    pub fn new(
        session_id: String,
        working_dir: Option<PathBuf>,
        tool_call_request_id: Option<String>,
    ) -> Self {
        Self {
            session_id,
            working_dir,
            tool_call_request_id,
            notification_emitter: None,
        }
    }

    pub fn working_dir_str(&self) -> Option<&str> {
        self.working_dir.as_ref().and_then(|p| p.to_str())
    }

    pub(crate) fn with_notification_emitter(
        mut self,
        notification_emitter: ToolCallNotificationEmitter,
    ) -> Self {
        self.notification_emitter = Some(notification_emitter);
        self
    }

    pub(crate) fn notification_emitter(&self) -> Option<&ToolCallNotificationEmitter> {
        self.notification_emitter.as_ref()
    }
}

// ToolCallResult combines the result of a tool call with an optional notification stream that
// can be used to receive notifications from the tool.
pub struct ToolCallResult {
    pub result: Box<dyn Future<Output = ToolResult<rmcp::model::CallToolResult>> + Send + Unpin>,
    pub notification_stream: Option<Box<dyn Stream<Item = ServerNotification> + Send + Unpin>>,
    pub action_required_stream: Option<Box<dyn Stream<Item = Message> + Send + Unpin>>,
}

impl From<ToolResult<rmcp::model::CallToolResult>> for ToolCallResult {
    fn from(result: ToolResult<rmcp::model::CallToolResult>) -> Self {
        Self {
            result: Box::new(futures::future::ready(result)),
            notification_stream: None,
            action_required_stream: None,
        }
    }
}

use crate::agents::Agent;
use crate::conversation::message::ToolRequest;
use crate::session::Session;
use crate::tool_inspection::get_security_finding_id_from_results;

pub(super) enum ToolStreamItem<T> {
    ActionRequired(Message),
    Message(ServerNotification),
    Result(T),
}

/// True for a tool confirmation a running tool passes up, a subagent's. It is answered live,
/// like the agent's own confirmations, so it is not kept in the conversation.
pub(super) fn is_tool_confirmation(message: &Message) -> bool {
    message.content.iter().any(|content| {
        matches!(
            content,
            MessageContent::ActionRequired(action)
                if matches!(action.data, ActionRequiredData::ToolConfirmation { .. })
        )
    })
}

pub(super) type ToolStream =
    Pin<Box<dyn Stream<Item = ToolStreamItem<ToolResult<CallToolResult>>> + Send>>;

pub(super) fn tool_stream<S, A, F>(rx: S, action_required_rx: A, done: F) -> ToolStream
where
    S: Stream<Item = ServerNotification> + Send + Unpin + 'static,
    A: Stream<Item = Message> + Send + Unpin + 'static,
    F: Future<Output = ToolResult<CallToolResult>> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        tokio::pin!(done);
        let mut rx = rx;
        let mut action_required_rx = action_required_rx;

        loop {
            tokio::select! {
                Some(msg) = action_required_rx.next() => {
                    yield ToolStreamItem::ActionRequired(msg);
                }
                Some(msg) = rx.next() => {
                    yield ToolStreamItem::Message(msg);
                }
                r = &mut done => {
                    yield ToolStreamItem::Result(r);
                    break;
                }
            }
        }
    })
}

pub const DECLINED_RESPONSE: &str = "The user has declined to run this tool. \
    DO NOT attempt to call this tool again. \
    If there are no alternative methods to proceed, clearly explain the situation and STOP.";

/// Why an ACP client refused a tool call, by session and tool request id (as
/// `ToolConfirmationRouter` keys a confirmation), kept until the refusal becomes the tool's result
/// or the session closes. A client's refusal is not always the user's: Intent's gate refuses some
/// calls by policy and says why, and "the user has declined" would then be untrue (centaur-core
/// #2106). Request ids are the model's and can repeat across sessions, so one session's reason
/// never answers another's call. Process-wide: the ACP server sets it, and the state machine's
/// operations, which hold no `Agent`, read it.
static CLIENT_REFUSAL_REASONS: LazyLock<Mutex<HashMap<(String, String), String>>> =
    LazyLock::new(Default::default);

fn client_refusal_reasons() -> MutexGuard<'static, HashMap<(String, String), String>> {
    CLIENT_REFUSAL_REASONS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

fn refusal_key(session_id: &str, request_id: &str) -> (String, String) {
    (session_id.to_string(), request_id.to_string())
}

pub fn record_client_refusal_reason(session_id: &str, request_id: &str, reason: String) {
    client_refusal_reasons().insert(refusal_key(session_id, request_id), reason);
}

pub fn forget_client_refusal_reason(session_id: &str, request_id: &str) {
    client_refusal_reasons().remove(&refusal_key(session_id, request_id));
}

/// Drops every reason recorded for a session, when it closes.
pub fn forget_client_refusal_reasons(session_id: &str) {
    client_refusal_reasons().retain(|(session, _), _| session != session_id);
}

/// Hands a reason the parent's client gave for a subagent's call to the subagent's session, whose
/// loop turns the refusal into the tool's result.
pub(crate) fn move_client_refusal_reason(from_session: &str, to_session: &str, request_id: &str) {
    let mut reasons = client_refusal_reasons();
    if let Some(reason) = reasons.remove(&refusal_key(from_session, request_id)) {
        reasons.insert(refusal_key(to_session, request_id), reason);
    }
}

/// What the model is told when a tool call was refused: the ACP client's reason if it gave one,
/// otherwise [`DECLINED_RESPONSE`].
pub fn declined_response(session_id: &str, request_id: &str) -> String {
    client_refusal_reasons()
        .remove(&refusal_key(session_id, request_id))
        .unwrap_or_else(|| DECLINED_RESPONSE.to_string())
}

pub const CHAT_MODE_TOOL_SKIPPED_RESPONSE: &str = "Let the user know the tool call was skipped in goose chat mode. \
                                        DO NOT apologize for skipping the tool call. DO NOT say sorry. \
                                        Provide an explanation of what the tool call would do, structured as a \
                                        plan for the user. Again, DO NOT apologize. \
                                        **Example Plan:**\n \
                                        1. **Identify Task Scope** - Determine the purpose and expected outcome.\n \
                                        2. **Outline Steps** - Break down the steps.\n \
                                        If needed, adjust the explanation based on user preferences or questions.";

impl Agent {
    pub(super) fn handle_approval_tool_requests<'a>(
        &'a self,
        tool_requests: &'a [ToolRequest],
        tool_futures: &'a mut Vec<(String, ToolStream)>,
        request_to_response_map: &'a mut HashMap<String, Message>,
        cancellation_token: Option<CancellationToken>,
        session: &'a Session,
        inspection_results: &'a [crate::tool_inspection::InspectionResult],
    ) -> BoxStream<'a, anyhow::Result<Message>> {
        try_stream! {
        for request in tool_requests.iter() {
            if let Ok(tool_call) = request.tool_call.clone() {
                let security_message = inspection_results.iter()
                    .find(|result| result.tool_request_id == request.id)
                    .and_then(|result| {
                        if let crate::tool_inspection::InspectionAction::RequireApproval(Some(message)) = &result.action {
                            Some(message.clone())
                        } else {
                            None
                        }
                    });

                let confirmation_rx = self
                    .tool_confirmation_router
                    .register(session.id.clone(), request.id.clone())
                    .await;

                let action_required_msg = Message::assistant()
                    .with_action_required(
                        request.id.clone(),
                        tool_call.name.to_string().clone(),
                        tool_call.arguments.clone().unwrap_or_default(),
                        security_message,
                    )
                    .user_only();
                yield action_required_msg;

                let confirmation = confirmation_rx.await
                    .map_err(|_| anyhow::anyhow!("Confirmation channel closed for request {}", request.id))?;

                if let Some(finding_id) = get_security_finding_id_from_results(&request.id, inspection_results) {
                    let action = match confirmation.permission {
                        Permission::AllowOnce | Permission::AlwaysAllow => "ALLOW",
                        _ => "BLOCK",
                    };
                    tracing::info!(
                        monotonic_counter.goose.prompt_injection_user_decisions = 1,
                        security.event_type = "user_decision",
                        security.action = action,
                        security.finding_id = %finding_id,
                        tool.request_id = %request.id,
                        user.decision = ?confirmation.permission,
                        "security finding: user decision"
                    );
                }

                if confirmation.permission == Permission::AllowOnce || confirmation.permission == Permission::AlwaysAllow {
                    let (req_id, tool_result) = self.dispatch_tool_call(tool_call.clone(), request.id.clone(), cancellation_token.clone(), session).await;

                    tool_futures.push((req_id, match tool_result {
                        Ok(result) => tool_stream(
                            result.notification_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.action_required_stream.unwrap_or_else(|| Box::new(stream::empty())),
                            result.result,
                        ),
                        Err(e) => tool_stream(
                            Box::new(stream::empty()),
                            Box::new(stream::empty()),
                            futures::future::ready(Err(e)),
                        ),
                    }));

                    if confirmation.permission == Permission::AlwaysAllow {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::AlwaysAllow)
                            .await;
                    }
                } else {
                    if let Some(response) = request_to_response_map.get_mut(&request.id) {
                        response.add_tool_response_with_metadata(
                            request.id.clone(),
                            Ok(CallToolResult::error(vec![ContentBlock::text(
                                declined_response(&session.id, &request.id),
                            )])),
                            request.metadata.as_ref(),
                        );
                    }

                    if confirmation.permission == Permission::AlwaysDeny {
                        self.tool_inspection_manager
                            .update_permission_manager(&tool_call.name, PermissionLevel::NeverAllow)
                            .await;
                    }
                }
            }
        }
    }.boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The map is process-wide and tests run in parallel, so each test uses its own session ids.

    #[test]
    fn a_reason_answers_only_its_own_sessions_call() {
        record_client_refusal_reason("same-id-a", "call_0", "policy".to_string());

        assert_eq!(declined_response("same-id-b", "call_0"), DECLINED_RESPONSE);
        assert_eq!(declined_response("same-id-a", "call_0"), "policy");
        assert_eq!(declined_response("same-id-a", "call_0"), DECLINED_RESPONSE);
    }

    #[test]
    fn closing_a_session_forgets_its_reasons_and_no_others() {
        record_client_refusal_reason("close-a", "call_0", "a0".to_string());
        record_client_refusal_reason("close-a", "call_1", "a1".to_string());
        record_client_refusal_reason("close-b", "call_0", "b0".to_string());

        forget_client_refusal_reasons("close-a");

        assert_eq!(declined_response("close-a", "call_0"), DECLINED_RESPONSE);
        assert_eq!(declined_response("close-a", "call_1"), DECLINED_RESPONSE);
        assert_eq!(declined_response("close-b", "call_0"), "b0");
    }

    #[test]
    fn a_reason_given_to_the_parent_reaches_the_subagents_refusal() {
        record_client_refusal_reason("parent", "sub_call", "policy".to_string());

        move_client_refusal_reason("parent", "subagent", "sub_call");

        assert_eq!(declined_response("parent", "sub_call"), DECLINED_RESPONSE);
        assert_eq!(declined_response("subagent", "sub_call"), "policy");
    }
}

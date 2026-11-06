use futures::{Stream, StreamExt};
use rmcp::model::CallToolResult;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};
use tokio::sync::mpsc;

use std::path::PathBuf;

use crate::agents::container::Container;
use crate::conversation::message::Message;
use crate::mcp_utils::ToolResult;
use rmcp::model::ServerNotification;

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
    container: Option<Container>,
    extension_lease: Option<Arc<ExtensionLease>>,
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
            container: None,
            extension_lease: None,
            notification_emitter: None,
        }
    }

    pub(crate) fn with_container(mut self, container: Option<Container>) -> Self {
        self.container = container;
        self
    }

    pub(crate) fn container(&self) -> Option<&Container> {
        self.container.as_ref()
    }

    pub(crate) fn with_extension_lease(mut self, extension_lease: Arc<ExtensionLease>) -> Self {
        self.extension_lease = Some(extension_lease);
        self
    }

    pub(crate) fn extension_lease(&self) -> Option<&Arc<ExtensionLease>> {
        self.extension_lease.as_ref()
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

use crate::agents::extension_manager::ExtensionLease;
use crate::agents::StateMachineServices;

pub(super) enum ToolStreamItem<T> {
    ActionRequired(Message),
    Message(ServerNotification),
    Result(T),
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

/// Why an ACP client refused a tool call, by session and tool request id, kept until the refusal
/// becomes the tool's result or the session ends. A client's refusal is not always the user's:
/// Intent's gate refuses some calls by policy and says why, and "the user has declined" would then
/// be untrue (centaur-core #2106). Request ids are the model's and can repeat across sessions, so
/// one session's reason never answers another's call.
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

pub fn take_client_refusal_reason(session_id: &str, request_id: &str) -> Option<String> {
    client_refusal_reasons().remove(&refusal_key(session_id, request_id))
}

/// Drops every reason recorded for a session, when it ends.
pub fn forget_client_refusal_reasons(session_id: &str) {
    client_refusal_reasons().retain(|(session, _), _| session != session_id);
}

pub const CHAT_MODE_TOOL_SKIPPED_RESPONSE: &str = "Let the user know the tool call was skipped in goose chat mode. \
                                        DO NOT apologize for skipping the tool call. DO NOT say sorry. \
                                        Provide an explanation of what the tool call would do, structured as a \
                                        plan for the user. Again, DO NOT apologize. \
                                        **Example Plan:**\n \
                                        1. **Identify Task Scope** - Determine the purpose and expected outcome.\n \
                                        2. **Outline Steps** - Break down the steps.\n \
                                        If needed, adjust the explanation based on user preferences or questions.";

impl StateMachineServices {}

#[cfg(test)]
mod client_refusal_reason_tests {
    use super::*;

    // The map is process-wide and tests run in parallel, so each test uses its own session ids.

    #[test]
    fn a_reason_answers_only_its_own_sessions_call_and_only_once() {
        record_client_refusal_reason("same-id-a", "call_0", "policy".to_string());

        assert_eq!(take_client_refusal_reason("same-id-b", "call_0"), None);
        assert_eq!(
            take_client_refusal_reason("same-id-a", "call_0").as_deref(),
            Some("policy")
        );
        assert_eq!(take_client_refusal_reason("same-id-a", "call_0"), None);
    }

    #[test]
    fn ending_a_session_forgets_its_reasons_and_no_others() {
        record_client_refusal_reason("close-a", "call_0", "a0".to_string());
        record_client_refusal_reason("close-a", "call_1", "a1".to_string());
        record_client_refusal_reason("close-b", "call_0", "b0".to_string());

        forget_client_refusal_reasons("close-a");

        assert_eq!(take_client_refusal_reason("close-a", "call_0"), None);
        assert_eq!(take_client_refusal_reason("close-a", "call_1"), None);
        assert_eq!(
            take_client_refusal_reason("close-b", "call_0").as_deref(),
            Some("b0")
        );
    }
}

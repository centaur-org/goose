//! An ACP client's refusal reason reaches the model in place of "the user declined" (#2106).
//!
//! Intent answers Goose's `session/request_permission` itself, and its gate refuses some calls by
//! policy (a send under "Ask me before nothing" to someone not named in the chat). Goose told the
//! model "The user has declined to run this tool" for every refusal, so the agent said the person
//! had declined when they had not. The fork patch `centaur: an ACP client's refusal reason reaches
//! the model in place of "the user declined"` reads the reason Intent puts in the answer,
//! `_meta.centaur.reason`, and gives it to the model as the tool's error text. Without a reason the
//! text is Goose's own, and an allow ignores the reason: only what a refusal says changes, never
//! what runs. Upstream has no way for a client to say why (aaif-goose/goose has no issue yet), so a
//! rebase that drops the patch makes the model hear "declined" again and fails here.
//!
//! The sync copies this file into the Goose checkout as `crates/goose/tests/centaur_refusal_reason.rs`.
//! It drives Goose's own ACP server in process with a scripted OpenAI-format model (Goose's test
//! fixtures): no key, no spend. The scripted model answers only a request that carries the text
//! the test expects the model to be told. Both agent loops are checked.

#![recursion_limit = "256"]
#[allow(dead_code, unused_imports)]
#[path = "acp_fixtures/mod.rs"]
mod fixtures;

use agent_client_protocol::schema::v1::Meta;
use fixtures::server::AcpServerConnection;
use fixtures::{
    run_test, Connection, OpenAiFixture, PermissionDecision, Session, SessionData,
    TestConnectionConfig,
};
use goose::agents::tool_execution::DECLINED_RESPONSE;
use goose::config::permission::PermissionLevel;
use goose::config::GooseMode;

const REASON: &str =
    "Under 'Ask me before nothing', Intent sends only to people named in this chat.";

#[test]
fn a_refusal_with_a_reason_tells_the_model_the_reason_legacy_loop() {
    run_test(async { assert_refusal(false, Case::PolicyRefusal).await });
}

#[test]
fn a_refusal_with_a_reason_tells_the_model_the_reason_state_machine() {
    run_test(async { assert_refusal(true, Case::PolicyRefusal).await });
}

#[test]
fn a_refusal_without_a_reason_still_says_the_user_declined_legacy_loop() {
    run_test(async { assert_refusal(false, Case::PlainRefusal).await });
}

#[test]
fn a_refusal_without_a_reason_still_says_the_user_declined_state_machine() {
    run_test(async { assert_refusal(true, Case::PlainRefusal).await });
}

#[test]
fn an_allow_with_a_reason_runs_the_tool_as_before_legacy_loop() {
    run_test(async { assert_refusal(false, Case::AllowWithReason).await });
}

#[test]
fn an_allow_with_a_reason_runs_the_tool_as_before_state_machine() {
    run_test(async { assert_refusal(true, Case::AllowWithReason).await });
}

#[test]
fn a_subagent_s_refused_call_tells_the_subagent_the_reason_legacy_loop() {
    run_test(async { assert_subagent_refusal(false).await });
}

#[test]
fn a_subagent_s_refused_call_tells_the_subagent_the_reason_state_machine() {
    run_test(async { assert_subagent_refusal(true).await });
}

#[derive(Clone, Copy, Debug)]
enum Case {
    PolicyRefusal,
    PlainRefusal,
    AllowWithReason,
}

/// Selects the agent loop until dropped; `run_test` holds the fixtures' lock meanwhile.
struct AgentLoop(Option<std::ffi::OsString>);

impl AgentLoop {
    fn select(state_machine: bool) -> Self {
        let previous = std::env::var_os("GOOSE_STATE_MACHINE");
        std::env::set_var("GOOSE_STATE_MACHINE", if state_machine { "1" } else { "0" });
        Self(previous)
    }
}

impl Drop for AgentLoop {
    fn drop(&mut self) {
        match self.0.take() {
            Some(previous) => std::env::set_var("GOOSE_STATE_MACHINE", previous),
            None => std::env::remove_var("GOOSE_STATE_MACHINE"),
        }
    }
}

fn openai_stream(delta: serde_json::Value, finish_reason: &str) -> &'static str {
    let chunk = |delta: serde_json::Value, finish_reason: Option<&str>| {
        serde_json::json!({
            "id": "chatcmpl-centaur-refusal",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5-nano",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })
    };
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk(delta, None),
        chunk(serde_json::json!({}), Some(finish_reason)),
    );
    Box::leak(body.into_boxed_str())
}

fn reason_meta() -> Meta {
    let serde_json::Value::Object(meta) = serde_json::json!({"centaur": {"reason": REASON}}) else {
        unreachable!()
    };
    meta
}

async fn assert_refusal(state_machine: bool, case: Case) {
    let _agent_loop = AgentLoop::select(state_machine);
    let scratch = tempfile::tempdir().unwrap();
    let marker = scratch.path().join("the-tool-ran");
    // The output, 42-ok, is not in the command, so only a model that got the tool's result sees it.
    let command = format!("touch {} && echo $((6*7))-ok", marker.display());
    // Each test its own call id: refusal reasons are kept by call id for the whole process.
    let call_id = format!(
        "call_refusal_{case:?}_{}",
        if state_machine { "sm" } else { "legacy" }
    );

    let (decision, meta, the_model_is_told) = match case {
        Case::PolicyRefusal => (PermissionDecision::RejectOnce, Some(reason_meta()), REASON),
        Case::PlainRefusal => (PermissionDecision::RejectOnce, None, DECLINED_RESPONSE),
        Case::AllowWithReason => (PermissionDecision::AllowOnce, Some(reason_meta()), "42-ok"),
    };

    let prompt = "Send the note.";
    let openai = OpenAiFixture::new(
        vec![
            (
                prompt.to_string(),
                openai_stream(
                    serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "shell",
                                     "arguments": serde_json::json!({"command": command}).to_string()},
                    }]}),
                    "tool_calls",
                ),
            ),
            (
                the_model_is_told.to_string(),
                openai_stream(
                    serde_json::json!({"role": "assistant", "content": "told"}),
                    "stop",
                ),
            ),
        ],
        AcpServerConnection::expected_session_id(),
    )
    .await;

    let config = TestConnectionConfig {
        builtins: vec!["developer".to_string()],
        goose_mode: GooseMode::Approve,
        ..Default::default()
    };
    let mut conn = AcpServerConnection::new(config, openai).await;
    conn.answer_permissions_with_meta(meta);
    let SessionData { mut session, .. } = conn.new_session().await.unwrap();
    let output = session.prompt(prompt, decision).await.unwrap();

    let asked = conn
        .permission_requests()
        .iter()
        .any(|request| request.tool_call.tool_call_id.0.as_ref() == call_id);
    assert!(
        asked,
        "{case:?}, state_machine={state_machine}: the client was asked about the call"
    );
    assert_eq!(
        output.text, "told",
        "{case:?}, state_machine={state_machine}: the model's next request carries {the_model_is_told:?}"
    );
    let ran = marker.exists();
    match case {
        Case::PolicyRefusal | Case::PlainRefusal => {
            assert!(
                !ran,
                "{case:?}, state_machine={state_machine}: a refused call ran"
            )
        }
        Case::AllowWithReason => {
            assert!(
                ran,
                "{case:?}, state_machine={state_machine}: an allowed call did not run"
            )
        }
    }
}

fn tool_call(call_id: &str, name: &str, arguments: serde_json::Value) -> &'static str {
    openai_stream(
        serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()},
        }]}),
        "tool_calls",
    )
}

fn text(text: &str) -> &'static str {
    openai_stream(
        serde_json::json!({"role": "assistant", "content": text}),
        "stop",
    )
}

/// A subagent's call is asked about on the parent's session, and the reason is kept by session,
/// so it has to reach the subagent's session for the subagent's model to hear it.
async fn assert_subagent_refusal(state_machine: bool) {
    let _agent_loop = AgentLoop::select(state_machine);
    let scratch = tempfile::tempdir().unwrap();
    let marker = scratch.path().join("the-subagent-ran-it");
    let command = format!("touch {}", marker.display());
    let suffix = if state_machine { "sm" } else { "legacy" };
    let parent_call = format!("call_refusal_delegate_{suffix}");
    let subagent_call = format!("call_refusal_subagent_{suffix}");

    let prompt = "Delegate sending the note.";
    let openai = OpenAiFixture::new(
        vec![
            (
                prompt.to_string(),
                tool_call(
                    &parent_call,
                    "delegate",
                    serde_json::json!({"instructions": format!("Run {command}")}),
                ),
            ),
            (
                "Subagent ID:".to_string(),
                tool_call(
                    &subagent_call,
                    "shell",
                    serde_json::json!({"command": command}),
                ),
            ),
            (REASON.to_string(), text("subagent told")),
            (parent_call.clone(), text("parent finished")),
        ],
        AcpServerConnection::expected_session_id(),
    )
    .await;

    let config = TestConnectionConfig {
        builtins: vec!["developer".to_string(), "summon".to_string()],
        goose_mode: GooseMode::Approve,
        ..Default::default()
    };
    let mut conn = AcpServerConnection::new(config, openai).await;
    conn.permission_manager()
        .update_user_permission("delegate", PermissionLevel::AlwaysAllow);
    conn.answer_permissions_with_meta(Some(reason_meta()));
    let SessionData { mut session, .. } = conn.new_session().await.unwrap();
    // The parent's delegate call is allowed without asking, so the one refusal is the subagent's.
    let output = session
        .prompt(prompt, PermissionDecision::RejectOnce)
        .await
        .unwrap();

    let asked: Vec<String> = conn
        .permission_requests()
        .iter()
        .map(|request| request.tool_call.tool_call_id.0.to_string())
        .collect();
    assert!(
        asked.contains(&subagent_call),
        "state_machine={state_machine}: the parent's client was asked about the subagent's call; \
         asked about: {asked:?}"
    );
    assert_eq!(
        output.text, "parent finished",
        "state_machine={state_machine}: the subagent's model was told {REASON:?}"
    );
    assert!(
        !marker.exists(),
        "state_machine={state_machine}: the refused call ran"
    );
}

use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::header::{HeaderName, HeaderValue};

pub const SESSION_ID_HEADER: &str = "agent-session-id";

pub const TOOL_CALL_REQUEST_ID_HEADER: &str = "agent-tool-call-request-id";
pub const WORKING_DIR_HEADER: &str = "agent-working-dir";

/// The `_meta` key of a tool call's proof, in the MCP `tools/call` Goose sends for it and (as
/// `proof`) in `_meta.goose.toolCall` of the ACP `tool_call` that reports it.
pub const TOOL_CALL_PROOF_HEADER: &str = "agent-tool-call-proof";
/// Set (to anything) to give each tool call a proof ([`tool_call_proof`]).
pub const TOOL_CALL_PROOF_ENV: &str = "GOOSE_TOOL_CALL_PROOF";

/// A key this process draws when it starts and never writes anywhere: not to its environment,
/// its config, its sessions or its logs.
static TOOL_CALL_PROOF_KEY: std::sync::LazyLock<Option<[u8; 32]>> =
    std::sync::LazyLock::new(|| {
        std::env::var_os(TOOL_CALL_PROOF_ENV).map(|_| rand::random::<[u8; 32]>())
    });

/// A proof that tool call `tool_call_request_id` is one this Goose made, for an ACP client that
/// serves an MCP's calls itself: Goose writes it only to its ACP client, in the call's
/// `tool_call`, and to the MCP it calls, in the `tools/call`. Its tool call id is also in the
/// session's files and the model's request log, which a process Goose's shell runs can read; the
/// proof is not, so a client that matches both knows the call came from Goose's MCP client and not
/// from such a process. HMAC-SHA256 of the id under [`TOOL_CALL_PROOF_KEY`], in hex. `None` unless
/// [`TOOL_CALL_PROOF_ENV`] is set.
pub fn tool_call_proof(tool_call_request_id: &str) -> Option<String> {
    TOOL_CALL_PROOF_KEY
        .as_ref()
        .map(|key| hmac_sha256_hex(key, tool_call_request_id.as_bytes()))
}

fn hmac_sha256_hex(key: &[u8; 32], message: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut inner_pad = [0x36u8; 64];
    let mut outer_pad = [0x5cu8; 64];
    for (index, byte) in key.iter().enumerate() {
        inner_pad[index] ^= byte;
        outer_pad[index] ^= byte;
    }
    let inner = Sha256::new()
        .chain_update(inner_pad)
        .chain_update(message)
        .finalize();
    Sha256::new()
        .chain_update(outer_pad)
        .chain_update(inner)
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

tokio::task_local! {
    pub static SESSION_ID: Option<String>;
}

pub async fn with_session_id<F>(session_id: Option<String>, f: F) -> F::Output
where
    F: std::future::Future,
{
    SESSION_ID.scope(session_id, f).await
}

pub fn with_session_id_stream<'a, T: Send + 'a>(
    session_id: Option<String>,
    stream: BoxStream<'a, T>,
) -> BoxStream<'a, T> {
    Box::pin(futures::stream::unfold(
        (stream, session_id),
        |(mut stream, session_id)| async move {
            with_session_id(session_id.clone(), stream.next())
                .await
                .map(|item| (item, (stream, session_id)))
        },
    ))
}

pub fn current_session_id() -> Option<String> {
    SESSION_ID.try_with(|id| id.clone()).ok().flatten()
}

pub fn session_id_request_builder() -> goose_providers::api_client::RequestBuilderDecorator {
    session_id_request_builder_with_header_name(HeaderName::from_static(SESSION_ID_HEADER))
}

pub(crate) fn session_id_request_builder_with_header_override(
    header_name_override: Option<&str>,
) -> Result<goose_providers::api_client::RequestBuilderDecorator, reqwest::header::InvalidHeaderName>
{
    let header_name = match header_name_override {
        Some(header_name) => HeaderName::from_bytes(header_name.as_bytes())?,
        None => HeaderName::from_static(SESSION_ID_HEADER),
    };

    Ok(session_id_request_builder_with_header_name(header_name))
}

fn session_id_request_builder_with_header_name(
    header_name: HeaderName,
) -> goose_providers::api_client::RequestBuilderDecorator {
    std::sync::Arc::new(move |request| {
        let (client, request) = request.build_split();
        let mut request = request?;
        let session_header = header_name.clone();
        request.headers_mut().remove(&session_header);

        if let Some(session_id) = current_session_id() {
            let value = HeaderValue::from_str(&session_id)?;
            request.headers_mut().insert(session_header, value);
        }

        Ok(reqwest::RequestBuilder::from_parts(client, request))
    })
}

/// Local OS user running goose, shared by the OTLP `user.name` resource
/// attribute and the `session.user` span attribute so the two never drift.
pub fn session_user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Hostname of the machine running goose, shared by the OTLP `host.name`
/// resource attribute and the `session.host` span attribute.
pub fn session_host() -> String {
    gethostname::gethostname().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_call_proof_is_hmac_sha256_of_its_id() {
        let key: [u8; 32] = std::array::from_fn(|index| index as u8);
        // Python: hmac.new(bytes(range(32)), b"toolu_1", hashlib.sha256).hexdigest()
        assert_eq!(
            hmac_sha256_hex(&key, b"toolu_1"),
            "4f701bcab1f6b13ed67972fbe0050849bc3121f3f09dd90fa3e2b6a36cde2245"
        );
        assert_ne!(
            hmac_sha256_hex(&key, b"toolu_1"),
            hmac_sha256_hex(&key, b"toolu_2")
        );
    }

    #[tokio::test]
    async fn test_session_id_available_when_set() {
        with_session_id(Some("test-session-123".to_string()), async {
            assert_eq!(current_session_id(), Some("test-session-123".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_when_not_set() {
        let id = current_session_id();
        assert_eq!(id, None);
    }

    #[tokio::test]
    async fn test_session_id_none_when_explicitly_none() {
        with_session_id(None, async {
            assert_eq!(current_session_id(), None);
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_none_clears_outer_scope() {
        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(None, async {
                assert_eq!(current_session_id(), None);
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_scoped_correctly() {
        assert_eq!(current_session_id(), None);

        with_session_id(Some("outer-session".to_string()), async {
            assert_eq!(current_session_id(), Some("outer-session".to_string()));

            with_session_id(Some("inner-session".to_string()), async {
                assert_eq!(current_session_id(), Some("inner-session".to_string()));
            })
            .await;

            assert_eq!(current_session_id(), Some("outer-session".to_string()));
        })
        .await;

        assert_eq!(current_session_id(), None);
    }

    #[tokio::test]
    async fn test_session_id_across_await_points() {
        with_session_id(Some("persistent-session".to_string()), async {
            assert_eq!(current_session_id(), Some("persistent-session".to_string()));

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            assert_eq!(current_session_id(), Some("persistent-session".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_scopes_each_stream_poll() {
        let stream = futures::stream::iter([(), ()]).then(|()| async { current_session_id() });
        let mut stream =
            with_session_id_stream(Some("stream-session".to_string()), Box::pin(stream));

        assert_eq!(
            stream.next().await,
            Some(Some("stream-session".to_string()))
        );
        assert_eq!(
            stream.next().await,
            Some(Some("stream-session".to_string()))
        );
        assert_eq!(stream.next().await, None);
        assert_eq!(current_session_id(), None);
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_custom_header() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate =
                session_id_request_builder_with_header_override(Some("x-opencode-session"))
                    .unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get("x-opencode-session").unwrap(),
                "test-session-123"
            );
            assert!(request.headers().get(SESSION_ID_HEADER).is_none());
        })
        .await;
    }

    #[tokio::test]
    async fn test_session_id_request_builder_uses_default_without_override() {
        with_session_id(Some("test-session-123".to_string()), async {
            let decorate = session_id_request_builder_with_header_override(None).unwrap();

            let request = decorate(reqwest::Client::new().get("http://localhost"))
                .unwrap()
                .build()
                .unwrap();

            assert_eq!(
                request.headers().get(SESSION_ID_HEADER).unwrap(),
                "test-session-123"
            );
        })
        .await;
    }

    #[test]
    fn test_session_id_request_builder_rejects_invalid_header_override() {
        assert!(session_id_request_builder_with_header_override(Some("invalid header")).is_err());
    }
}

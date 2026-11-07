//! The HTTP abstraction. Everything that touches the network goes through [`Fetch`].

use std::io::Write;

use crate::Result;

/// Minimal HTTP GET interface. Implementations must return [`crate::Error::Status`] for non-2xx
/// responses so callers can tell "missing tile" (404) from transport failures.
pub trait Fetch: Sync {
    /// GET `url` and stream the body into `out`. Returns the number of bytes written.
    fn get_to(&self, url: &str, out: &mut dyn Write) -> Result<u64>;

    /// GET `url` into memory. For small payloads (tiles, checksums, the region index).
    fn get(&self, url: &str) -> Result<Vec<u8>> {
        let mut v = Vec::new();
        self.get_to(url, &mut v)?;
        Ok(v)
    }
}

/// Blocking HTTPS client (ureq + rustls, Mozilla roots via webpki-roots).
#[cfg(feature = "net")]
pub struct HttpFetcher {
    agent: ureq::Agent,
}

#[cfg(feature = "net")]
impl Default for HttpFetcher {
    fn default() -> Self {
        let config = ureq::Agent::config_builder()
            .user_agent(crate::USER_AGENT)
            .timeout_connect(Some(std::time::Duration::from_secs(20)))
            .timeout_recv_body(None) // PBF extracts can be gigabytes; rely on per-read progress
            .http_status_as_error(false)
            .build();
        Self { agent: config.into() }
    }
}

#[cfg(feature = "net")]
impl Fetch for HttpFetcher {
    fn get_to(&self, url: &str, out: &mut dyn Write) -> Result<u64> {
        use crate::Error;
        let net = |e: &dyn std::fmt::Display| Error::Net { url: url.to_string(), msg: e.to_string() };
        let resp = self.agent.get(url).call().map_err(|e| net(&e))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            return Err(Error::Status { url: url.to_string(), status });
        }
        let mut body = resp.into_body();
        let mut reader = body.with_config().limit(u64::MAX).reader();
        std::io::copy(&mut reader, out).map_err(|e| net(&e))
    }
}

#[cfg(test)]
pub(crate) mod mock {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
    use crate::Error;

    /// In-memory fetcher: serves canned bodies, 404s everything else, records requests.
    #[derive(Default)]
    pub struct MockFetcher {
        pub bodies: HashMap<String, Vec<u8>>,
        pub log: Mutex<Vec<String>>,
    }

    impl MockFetcher {
        pub fn with(mut self, url: &str, body: impl Into<Vec<u8>>) -> Self {
            self.bodies.insert(url.to_string(), body.into());
            self
        }
        pub fn requests(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    impl Fetch for MockFetcher {
        fn get_to(&self, url: &str, out: &mut dyn Write) -> Result<u64> {
            self.log.lock().unwrap().push(url.to_string());
            match self.bodies.get(url) {
                Some(b) => {
                    out.write_all(b).unwrap();
                    Ok(b.len() as u64)
                }
                None => Err(Error::Status { url: url.to_string(), status: 404 }),
            }
        }
    }
}

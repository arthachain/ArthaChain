use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInstant {
    #[serde(with = "serde_instant")]
    pub instant: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDuration {
    #[serde(with = "serde_duration")]
    pub duration: Duration,
}

impl SerializableInstant {
    pub fn now() -> Self {
        Self { instant: Instant::now() }
    }

    pub fn elapsed(&self) -> Duration {
        self.instant.elapsed()
    }
}

mod serde_instant {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, Instant};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = instant.duration_since(Instant::now());
        serializer.serialize_i64(duration.as_secs() as i64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = i64::deserialize(deserializer)?;
        Ok(Instant::now() + Duration::from_secs(secs as u64))
    }
}

mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(duration.as_secs() as i64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = i64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs as u64))
    }
} 
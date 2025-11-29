use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Board information from /board endpoint
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BoardInfo {
    pub board_connected: bool,
    pub board_type: String,
    pub num_channels: u8,
    pub gains: Vec<u8>,
}

/// All shield information from /all endpoint
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ShieldInfo {
    pub board_connected: bool,
    pub heap: u32,
    pub ip: String,
    pub mac: String,
    pub name: String,
    pub num_channels: u8,
    pub version: String,
    pub latency: u32,
}

/// TCP streaming configuration
#[derive(Debug, Serialize)]
pub struct TcpConfig {
    pub ip: String,
    pub port: u16,
    pub output: String, // "json" or "raw"
    pub delimiter: bool,
    pub latency: u32, // microseconds between packets
    #[serde(skip_serializing_if = "Option::is_none")]
    pub burst: Option<bool>,
}

/// OpenBCI WiFi Shield client
pub struct OpenBCIWiFi {
    ip_address: String,
    client: Client,
}

impl OpenBCIWiFi {
    /// Create a new OpenBCI WiFi Shield client
    pub fn new(ip_address: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            ip_address: ip_address.to_string(),
            client,
        }
    }

    /// Get board information
    pub async fn get_board_info(&self) -> Result<BoardInfo> {
        let url = format!("http://{}/board", self.ip_address);
        debug!("Fetching board info from {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?;

        let board_info: BoardInfo = response
            .json()
            .await
            .context("Failed to parse board info")?;

        Ok(board_info)
    }

    /// Get all shield information
    pub async fn get_shield_info(&self) -> Result<ShieldInfo> {
        let url = format!("http://{}/all", self.ip_address);
        debug!("Fetching shield info from {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?;

        let shield_info: ShieldInfo = response
            .json()
            .await
            .context("Failed to parse shield info")?;

        Ok(shield_info)
    }

    /// Get firmware version
    pub async fn get_version(&self) -> Result<String> {
        let url = format!("http://{}/version", self.ip_address);
        debug!("Fetching version from {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?;

        let text = response.text().await?;
        Ok(text)
    }

    /// Start TCP streaming
    pub async fn start_tcp_stream(
        &self,
        local_ip: &str,
        local_port: u16,
        output_format: &str,
        latency_us: u32,
    ) -> Result<()> {
        let config = TcpConfig {
            ip: local_ip.to_string(),
            port: local_port,
            output: output_format.to_string(),
            delimiter: true,
            latency: latency_us,
            burst: Some(false),
        };

        let url = format!("http://{}/tcp", self.ip_address);
        info!("Starting TCP stream to {}:{}", local_ip, local_port);
        debug!("TCP config: {:?}", config);

        let response = self
            .client
            .post(&url)
            .json(&config)
            .send()
            .await
            .context("Failed to start TCP stream")?;

        if response.status().is_success() {
            info!("TCP stream started successfully");
            Ok(())
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("Failed to start TCP stream: {} - {}", status, text);
            anyhow::bail!("Failed to start TCP stream: {}", status)
        }
    }

    /// Stop streaming
    pub async fn stop_stream(&self) -> Result<()> {
        let url = format!("http://{}/tcp", self.ip_address);
        info!("Stopping TCP stream");

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .context("Failed to stop stream")?;

        if response.status().is_success() {
            info!("Stream stopped successfully");
            Ok(())
        } else {
            warn!("Failed to stop stream: {}", response.status());
            Ok(()) // Don't fail on stop errors
        }
    }

    /// Send a command to the board
    pub async fn send_command(&self, command: &str) -> Result<String> {
        let url = format!("http://{}/command", self.ip_address);
        info!("Sending command: {}", command);

        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "command": command }))
            .send()
            .await
            .context("Failed to send command")?;

        let text = response.text().await?;
        Ok(text)
    }

    /// Get the IP address of this shield
    pub fn ip_address(&self) -> &str {
        &self.ip_address
    }
}

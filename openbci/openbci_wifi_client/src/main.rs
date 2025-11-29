use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

/// Board information from /board endpoint
#[derive(Debug, Deserialize, Serialize)]
struct BoardInfo {
    board_connected: bool,
    board_type: String,
    num_channels: u8,
    gains: Vec<u8>,
}

/// All shield information from /all endpoint
#[derive(Debug, Deserialize, Serialize)]
struct ShieldInfo {
    board_connected: bool,
    heap: u32,
    ip: String,
    mac: String,
    name: String,
    num_channels: u8,
    version: String,
    latency: u32,
}

/// TCP streaming configuration
#[derive(Debug, Serialize)]
struct TcpConfig {
    ip: String,
    port: u16,
    output: String,      // "json" or "raw"
    delimiter: bool,
    latency: u32,        // microseconds between packets
    #[serde(skip_serializing_if = "Option::is_none")]
    burst: Option<bool>,
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
        info!("Fetching board info from {}", url);

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
        info!("Fetching shield info from {}", url);

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
        info!("Fetching version from {}", url);

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
}

/// TCP data receiver
pub struct TcpDataReceiver {
    port: u16,
}

impl TcpDataReceiver {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    /// Start listening for data
    pub async fn listen<F>(&self, callback: F) -> Result<()>
    where
        F: FnMut(Vec<u8>) + Send + 'static,
    {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr)
            .await
            .context(format!("Failed to bind to {}", addr))?;

        info!("TCP listener started on {}", addr);

        let callback = Arc::new(Mutex::new(callback));

        loop {
            match listener.accept().await {
                Ok((mut socket, addr)) => {
                    info!("New connection from: {}", addr);

                    let callback_clone = Arc::clone(&callback);
                    tokio::spawn(async move {
                        let mut buffer = vec![0u8; 8192];

                        loop {
                            match socket.read(&mut buffer).await {
                                Ok(0) => {
                                    info!("Connection closed by {}", addr);
                                    break;
                                }
                                Ok(n) => {
                                    debug!("Received {} bytes from {}", n, addr);
                                    let mut cb = callback_clone.lock().unwrap();
                                    cb(buffer[..n].to_vec());
                                }
                                Err(e) => {
                                    error!("Error reading from socket: {}", e);
                                    break;
                                }
                            }
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("OpenBCI WiFi Shield Client - Starting");

    // Create client
    let shield = OpenBCIWiFi::new("192.168.4.1");

    // Test connection
    info!("Testing connection to OpenBCI WiFi Shield...");

    match shield.get_board_info().await {
        Ok(board_info) => {
            info!("Board Info:");
            info!("  Connected: {}", board_info.board_connected);
            info!("  Type: {}", board_info.board_type);
            info!("  Channels: {}", board_info.num_channels);
            info!("  Gains: {:?}", board_info.gains);
        }
        Err(e) => {
            error!("Failed to get board info: {}", e);
            error!("Make sure you're connected to OpenBCI-E324 WiFi network");
            return Err(e);
        }
    }

    // Get shield info
    match shield.get_shield_info().await {
        Ok(info) => {
            info!("Shield Info:");
            info!("  Name: {}", info.name);
            info!("  Version: {}", info.version);
            info!("  MAC: {}", info.mac);
            info!("  Heap: {} bytes", info.heap);
            info!("  Latency: {} us", info.latency);
        }
        Err(e) => {
            warn!("Failed to get shield info: {}", e);
        }
    }

    // Get version
    match shield.get_version().await {
        Ok(version) => {
            info!("Firmware version: {}", version);
        }
        Err(e) => {
            warn!("Failed to get version: {}", e);
        }
    }

    info!("\n=== Connection Test Successful! ===\n");

    // Example: Start streaming (commented out)
    // Uncomment to test data streaming
    
    info!("Starting data stream...");
    
    // Get local IP on wlan1
    let local_ip = "192.168.4.2"; // Your laptop's IP on OpenBCI network
    let local_port = 3000;
    
    // Start TCP listener in background
    let receiver = TcpDataReceiver::new(local_port);
    tokio::spawn(async move {
        receiver.listen(|data| {
            info!("Received {} bytes", data.len());
            // Process data here
        }).await
    });
    
    // Wait a bit for listener to start
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Start streaming from shield
    shield.start_tcp_stream(local_ip, local_port, "json", 10000).await?;
    
    info!("Streaming for 10 seconds...");
    tokio::time::sleep(Duration::from_secs(10)).await;
    
    // Stop streaming
    shield.stop_stream().await?;


    info!("Test complete!");

    Ok(())
}

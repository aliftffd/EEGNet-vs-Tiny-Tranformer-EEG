/// Example: Stream OpenBCI data continuously
/// Run with: cargo run --example stream
use anyhow::Result;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

mod openbci {
    pub use openbci_wifi_client::*;
}

/// JSON format data from OpenBCI WiFi Shield
#[derive(Debug, Deserialize, Serialize)]
struct OpenBCIChunk {
    chunk: Vec<OpenBCISample>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenBCISample {
    data: Vec<f32>,      // Channel data in nanovolts
    timestamp: f64,      // Timestamp
}

async fn stream_data(shield_ip: &str, local_ip: &str, local_port: u16) -> Result<()> {
    info!("Starting OpenBCI data stream");

    // Create HTTP client for control
    let client = reqwest::Client::new();

    // Start TCP listener
    let listener_addr = format!("0.0.0.0:{}", local_port);
    let listener = TcpListener::bind(&listener_addr).await?;
    info!("Listening on {}", listener_addr);

    // Start streaming from shield
    let tcp_config = serde_json::json!({
        "ip": local_ip,
        "port": local_port,
        "output": "json",
        "delimiter": true,
        "latency": 10000,
        "burst": false
    });

    let url = format!("http://{}/tcp", shield_ip);
    info!("Starting stream from shield at {}", url);

    let response = client.post(&url).json(&tcp_config).send().await?;

    if !response.status().is_success() {
        error!("Failed to start stream: {}", response.status());
        return Ok(());
    }

    info!("Stream started successfully");

    // Accept connection
    let (mut socket, addr) = listener.accept().await?;
    info!("Connected to: {}", addr);

    let mut buffer = vec![0u8; 16384];
    let mut sample_count = 0;

    loop {
        match socket.read(&mut buffer).await {
            Ok(0) => {
                info!("Connection closed");
                break;
            }
            Ok(n) => {
                // Try to parse JSON
                let data_str = String::from_utf8_lossy(&buffer[..n]);
                
                // Split by delimiter if present
                for line in data_str.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    
                    match serde_json::from_str::<OpenBCIChunk>(line) {
                        Ok(chunk) => {
                            for sample in chunk.chunk {
                                sample_count += 1;
                                
                                if sample_count % 100 == 0 {
                                    info!(
                                        "Sample {}: {} channels, timestamp: {:.3}",
                                        sample_count,
                                        sample.data.len(),
                                        sample.timestamp
                                    );
                                    
                                    // Show first 4 channel values
                                    let preview: Vec<f32> = sample.data.iter().take(4).copied().collect();
                                    info!("  Channel preview: {:?}", preview);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to parse JSON: {} - Data: {}", e, line);
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error reading from socket: {}", e);
                break;
            }
        }
    }

    // Stop streaming
    let stop_url = format!("http://{}/tcp", shield_ip);
    let _ = client.delete(&stop_url).send().await;

    info!("Total samples received: {}", sample_count);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let shield_ip = "192.168.4.1";
    
    // Get your laptop's IP on wlan1 (OpenBCI network)
    let local_ip = "192.168.4.2"; // Adjust if different
    let local_port = 3000;

    info!("OpenBCI WiFi Streaming Example");
    info!("Shield IP: {}", shield_ip);
    info!("Local IP: {}", local_ip);
    info!("Local Port: {}", local_port);
    info!("\nPress Ctrl+C to stop\n");

    tokio::time::sleep(Duration::from_secs(1)).await;

    stream_data(shield_ip, local_ip, local_port).await?;

    Ok(())
}

use anyhow::Result;
use chrono::{DateTime, Utc};
use clap::Parser;
use log::{error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "OpenBCI Motor Imagery Data Collector")]
#[command(about = "Collect and save OpenBCI EEG data for motor imagery deep learning", long_about = None)]
struct Args {
    /// OpenBCI WiFi Shield IP address
    #[arg(short, long, default_value = "192.168.4.1")]
    shield_ip: String,

    /// Local IP address (your laptop on wlan1)
    #[arg(short, long, default_value = "192.168.4.2")]
    local_ip: String,

    /// TCP port for data reception
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Output directory for saved data
    #[arg(short, long, default_value = "motor_imagery_data")]
    output_dir: String,

    /// Motor imagery class: left_hand, right_hand, both_hands, rest
    #[arg(short = 'c', long)]
    class: String,

    /// Trial number (for organizing multiple repetitions)
    #[arg(short = 't', long, default_value = "1")]
    trial: u32,

    /// Duration per trial in seconds
    #[arg(short, long, default_value = "5")]
    duration: u64,

    /// Sampling rate (Hz)
    #[arg(short = 'r', long, default_value = "250")]
    sample_rate: u32,

    /// Number of channels to record
    #[arg(long, default_value = "2")]
    channels: usize,

    /// Subject ID
    #[arg(long, default_value = "S01")]
    subject_id: String,

    /// Session ID (for grouping trials in one recording session)
    #[arg(long, default_value = "session_01")]
    session_id: String,
}

/// EEG sample with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EEGSample {
    timestamp: f64,
    sample_id: u64,
    channels: Vec<f32>,
}

/// Motor imagery trial metadata
#[derive(Debug, Serialize, Deserialize)]
struct TrialMetadata {
    subject_id: String,
    session_id: String,
    trial_number: u32,
    class_label: String,
    class_id: u8,
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    sample_rate: u32,
    num_channels: usize,
    total_samples: u64,
    duration_seconds: u64,
    electrode_config: ElectrodeConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct ElectrodeConfig {
    channels: Vec<String>,
    reference: String,
    ground: String,
}

/// Map motor imagery class names to numeric IDs for deep learning
fn get_class_id(class_name: &str) -> u8 {
    match class_name.to_lowercase().as_str() {
        "left_hand" | "left" => 0,
        "right_hand" | "right" => 1,
        "both_hands" | "both" => 2,
        "rest" | "baseline" => 3,
        _ => {
            warn!("Unknown class '{}', defaulting to rest (3)", class_name);
            3
        }
    }
}

/// Data buffer for batch writing
struct DataBuffer {
    samples: Vec<EEGSample>,
    capacity: usize,
}

impl DataBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, sample: EEGSample) -> bool {
        self.samples.push(sample);
        self.samples.len() >= self.capacity
    }

    fn clear(&mut self) -> Vec<EEGSample> {
        std::mem::take(&mut self.samples)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Data writer for CSV format
struct CSVWriter {
    file_path: PathBuf,
    writer: csv::Writer<std::fs::File>,
    samples_written: u64,
    class_id: u8,
}

impl CSVWriter {
    fn new(output_dir: &str, subject_id: &str, session_id: &str, class_label: &str, trial: u32, class_id: u8, num_channels: usize) -> Result<Self> {
        // Create directory structure: motor_imagery_data/S01/session_01/
        let subject_dir = PathBuf::from(output_dir).join(subject_id).join(session_id);
        fs::create_dir_all(&subject_dir)?;

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        // Filename: S01_left_hand_trial_01_class_0_20250128_143022.csv
        let filename = format!("{}_{}_{}_trial_{:02}_class_{}_{}.csv",
                              subject_id, class_label, session_id, trial, class_id, timestamp);
        let file_path = subject_dir.join(filename);

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)?;

        let mut writer = csv::Writer::from_writer(file);

        // Generate channel labels with motor cortex annotations
        let channel_labels = Self::generate_channel_labels(num_channels);

        // Write header with class_id for easy loading in deep learning
        let mut header = vec!["timestamp".to_string(), "sample_id".to_string(), "class_id".to_string()];
        header.extend(channel_labels.clone());
        writer.write_record(&header)?;

        Ok(Self {
            file_path,
            writer,
            samples_written: 0,
            class_id,
        })
    }

    fn generate_channel_labels(num_channels: usize) -> Vec<String> {
        // Map channels to standard 10-20 positions with motor cortex labels
        let labels = vec![
            "C3_left_motor",
            "C4_right_motor",
            "Cz_central",
            "F3_frontal_left",
            "F4_frontal_right",
            "P3_parietal_left",
            "P4_parietal_right",
            "O1_occipital_left",
        ];

        labels.iter()
            .take(num_channels)
            .map(|s| s.to_string())
            .collect()
    }

    fn write_batch(&mut self, samples: &[EEGSample]) -> Result<()> {
        for sample in samples {
            let mut record = vec![
                sample.timestamp.to_string(),
                sample.sample_id.to_string(),
                self.class_id.to_string(),
            ];
            for ch in &sample.channels {
                record.push(ch.to_string());
            }
            self.writer.write_record(&record)?;
            self.samples_written += 1;
        }

        self.writer.flush()?;
        info!("Wrote {} samples to CSV (total: {})", samples.len(), self.samples_written);

        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        info!("Finalized CSV file: {:?}", self.file_path);
        Ok(())
    }
}

/// Main data collector
struct DataCollector {
    shield_ip: String,
    local_ip: String,
    port: u16,
    client: Client,
    buffer: Arc<Mutex<DataBuffer>>,
    csv_writer: Arc<Mutex<CSVWriter>>,
    metadata: TrialMetadata,
    sample_count: Arc<Mutex<u64>>,
    start_time: Instant,
}

impl DataCollector {
    fn new(args: &Args) -> Result<Self> {
        // Create output directory
        fs::create_dir_all(&args.output_dir)?;

        // Generate channel labels matching CSV headers
        let channel_names = CSVWriter::generate_channel_labels(args.channels);

        let electrode_config = ElectrodeConfig {
            channels: channel_names,
            reference: "Cz".to_string(),
            ground: "Fpz".to_string(),
        };

        let class_id = get_class_id(&args.class);

        let metadata = TrialMetadata {
            subject_id: args.subject_id.clone(),
            session_id: args.session_id.clone(),
            trial_number: args.trial,
            class_label: args.class.clone(),
            class_id,
            start_time: Utc::now(),
            end_time: None,
            sample_rate: args.sample_rate,
            num_channels: args.channels,
            total_samples: 0,
            duration_seconds: args.duration,
            electrode_config,
        };

        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;

        let buffer = Arc::new(Mutex::new(DataBuffer::new(250))); // Buffer 1 second at 250Hz

        let csv_writer = Arc::new(Mutex::new(CSVWriter::new(
            &args.output_dir,
            &args.subject_id,
            &args.session_id,
            &args.class,
            args.trial,
            class_id,
            args.channels,
        )?));

        Ok(Self {
            shield_ip: args.shield_ip.clone(),
            local_ip: args.local_ip.clone(),
            port: args.port,
            client,
            buffer,
            csv_writer,
            metadata,
            sample_count: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        })
    }

    async fn start_streaming(&self) -> Result<()> {
        // First, try to stop any existing TCP stream
        info!("Cleaning up any existing TCP streams");
        let stop_url = format!("http://{}/tcp", self.shield_ip);
        let _ = self.client.delete(&stop_url).send().await;

        // Wait a moment for cleanup
        tokio::time::sleep(Duration::from_millis(500)).await;

        let tcp_config = serde_json::json!({
            "ip": self.local_ip,
            "port": self.port,
            "output": "json",
            "delimiter": true,
            "latency": 4000, // 4ms for 250Hz
            "burst": false
        });

        let url = format!("http://{}/tcp", self.shield_ip);
        info!("Starting TCP stream from {}", url);
        info!("Config: ip={}, port={}", self.local_ip, self.port);

        // Use a longer timeout specifically for this request (30 seconds)
        let client_with_long_timeout = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;

        let response = client_with_long_timeout
            .post(&url)
            .json(&tcp_config)
            .send()
            .await?;

        if response.status().is_success() {
            let body = response.text().await?;
            info!("Stream started successfully. Response: {}", body);
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "Unable to read response body".to_string());
            anyhow::bail!("Failed to start stream: {} - {}", status, body)
        }
    }

    async fn stop_streaming(&self) -> Result<()> {
        let url = format!("http://{}/tcp", self.shield_ip);
        info!("Stopping stream");
        let _ = self.client.delete(&url).send().await;
        Ok(())
    }

    async fn collect_data(&mut self, duration_secs: u64) -> Result<()> {
        info!("Starting data collection for {} seconds", duration_secs);
        let channels_str = self.metadata.electrode_config.channels.join(", ");
        info!("Electrode configuration: {} (active) | {} (ref) | {} (gnd)",
              channels_str,
              self.metadata.electrode_config.reference,
              self.metadata.electrode_config.ground);

        // Setup TCP listener FIRST (before starting stream)
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        info!("Listening on {}", addr);

        // Now start streaming (this will cause the board to connect to us)
        self.start_streaming().await?;

        // Accept connection with timeout
        let accept_future = listener.accept();
        let (mut socket, addr) = tokio::time::timeout(
            Duration::from_secs(10),
            accept_future
        ).await??;
        
        info!("Connected to: {}", addr);

        let mut buffer_vec = vec![0u8; 16384];
        let end_time = if duration_secs > 0 {
            Some(Instant::now() + Duration::from_secs(duration_secs))
        } else {
            None
        };

        let sample_count = Arc::clone(&self.sample_count);
        let buffer = Arc::clone(&self.buffer);
        let csv_writer = Arc::clone(&self.csv_writer);

        let mut last_progress = Instant::now();

        loop {
            // Check if we should stop
            if let Some(end) = end_time {
                if Instant::now() >= end {
                    info!("Duration reached, stopping collection");
                    break;
                }
            }

            // Read data with timeout
            let read_future = socket.read(&mut buffer_vec);
            match tokio::time::timeout(Duration::from_millis(100), read_future).await {
                Ok(Ok(0)) => {
                    warn!("Connection closed");
                    break;
                }
                Ok(Ok(n)) => {
                    let data_str = String::from_utf8_lossy(&buffer_vec[..n]);
                    
                    for line in data_str.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        
                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(line) {
                            if let Some(samples_array) = chunk.get("chunk").and_then(|c| c.as_array()) {
                                for sample_json in samples_array {
                                    if let (Some(data), Some(ts)) = (
                                        sample_json.get("data").and_then(|d| d.as_array()),
                                        sample_json.get("timestamp").and_then(|t| t.as_f64())
                                    ) {
                                        let channels: Vec<f32> = data
                                            .iter()
                                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                                            .collect();

                                        let mut count = sample_count.lock().unwrap();
                                        let sample = EEGSample {
                                            timestamp: ts,
                                            sample_id: *count,
                                            channels,
                                        };
                                        *count += 1;

                                        let mut buf = buffer.lock().unwrap();
                                        if buf.push(sample) {
                                            // Buffer full, write to disk
                                            let samples_to_write = buf.clear();

                                            let mut w = csv_writer.lock().unwrap();
                                            if let Err(e) = w.write_batch(&samples_to_write) {
                                                error!("Failed to write to CSV: {}", e);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Progress update every 5 seconds
                    if last_progress.elapsed() >= Duration::from_secs(5) {
                        let count = *sample_count.lock().unwrap();
                        let elapsed = self.start_time.elapsed().as_secs();
                        let rate = count as f64 / elapsed as f64;
                        info!("Collected {} samples ({:.1} Hz)", count, rate);
                        last_progress = Instant::now();
                    }
                }
                Ok(Err(e)) => {
                    error!("Error reading: {}", e);
                    break;
                }
                Err(_) => {
                    // Timeout, continue
                }
            }
        }

        // Write remaining buffered samples
        let mut buf = buffer.lock().unwrap();
        if buf.len() > 0 {
            let samples_to_write = buf.clear();

            let mut w = csv_writer.lock().unwrap();
            let _ = w.write_batch(&samples_to_write);
        }

        self.stop_streaming().await?;

        Ok(())
    }

    fn finalize(&mut self, output_dir: &str) -> Result<()> {
        let total_samples = *self.sample_count.lock().unwrap();
        self.metadata.end_time = Some(Utc::now());
        self.metadata.total_samples = total_samples;

        info!("Finalizing data collection...");
        info!("Total samples collected: {}", total_samples);

        let mut w = self.csv_writer.lock().unwrap();
        w.finalize()?;

        // Save metadata in same directory structure as CSV
        let subject_dir = PathBuf::from(output_dir)
            .join(&self.metadata.subject_id)
            .join(&self.metadata.session_id);

        let metadata_filename = format!("{}_{}_trial_{:02}_class_{}_metadata.json",
                                       self.metadata.subject_id,
                                       self.metadata.class_label,
                                       self.metadata.trial_number,
                                       self.metadata.class_id);
        let metadata_path = subject_dir.join(metadata_filename);
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        info!("Saved metadata to: {:?}", metadata_path);

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args = Args::parse();

    info!("=== OpenBCI Motor Imagery Data Collector ===");
    info!("Subject: {}", args.subject_id);
    info!("Session: {}", args.session_id);
    info!("Class: {} (ID: {})", args.class, get_class_id(&args.class));
    info!("Trial: {}", args.trial);
    info!("Duration: {} seconds", args.duration);
    info!("Output: {}", args.output_dir);
    info!("Channels: {}", args.channels);
    info!("");

    let mut collector = DataCollector::new(&args)?;

    match collector.collect_data(args.duration).await {
        Ok(_) => {
            info!("Data collection completed successfully");
        }
        Err(e) => {
            error!("Error during collection: {}", e);
        }
    }

    collector.finalize(&args.output_dir)?;

    info!("=== Collection Complete ===");

    Ok(())
}

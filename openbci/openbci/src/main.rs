use esp_idf_hal::{
    gpio::{Gpio0, Gpio2, Gpio12, Gpio13, Gpio14, Gpio15, PinDriver},
    peripherals::Peripherals,
    prelude::*,
};
use esp_idf_svc::{
    eventloop::EspSystemEventLoop,
    nvs::EspDefaultNvsPartition,
    wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi},
};
use esp_idf_sys as _;
use log::*;
use rosc::{OscMessage, OscPacket, OscType};
use std::{
    net::UdpSocket,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

// Settings
const WIFI_SSID: &str = "**YOUR SSID**";
const WIFI_PASSWORD: &str = "**YOUR PASSWORD**";
const LISTEN_PORT: u16 = 9002;
const TIMEOUT_MILLIS: u64 = 5000;

// Motor pin definitions
struct MotorPins<'a> {
    left_forward: PinDriver<'a, Gpio2, esp_idf_hal::gpio::Output>,
    right_forward: PinDriver<'a, Gpio15, esp_idf_hal::gpio::Output>,
    left_backward: PinDriver<'a, Gpio0, esp_idf_hal::gpio::Output>,
    right_backward: PinDriver<'a, Gpio13, esp_idf_hal::gpio::Output>,
    left_enable: PinDriver<'a, Gpio12, esp_idf_hal::gpio::Output>,
    right_enable: PinDriver<'a, Gpio14, esp_idf_hal::gpio::Output>,
}

impl<'a> MotorPins<'a> {
    fn new(peripherals: &'a mut Peripherals) -> anyhow::Result<Self> {
        Ok(Self {
            left_forward: PinDriver::output(peripherals.pins.gpio2.downgrade_output())?,
            right_forward: PinDriver::output(peripherals.pins.gpio15.downgrade_output())?,
            left_backward: PinDriver::output(peripherals.pins.gpio0.downgrade_output())?,
            right_backward: PinDriver::output(peripherals.pins.gpio13.downgrade_output())?,
            left_enable: PinDriver::output(peripherals.pins.gpio12.downgrade_output())?,
            right_enable: PinDriver::output(peripherals.pins.gpio14.downgrade_output())?,
        })
    }

    /// Move forward
    fn forward(&mut self) -> anyhow::Result<()> {
        self.left_enable.set_high()?;
        self.right_enable.set_high()?;
        self.left_forward.set_high()?;
        self.right_forward.set_high()?;
        self.left_backward.set_low()?;
        self.right_backward.set_low()?;
        Ok(())
    }

    /// Move backward
    fn backward(&mut self) -> anyhow::Result<()> {
        self.left_enable.set_high()?;
        self.right_enable.set_high()?;
        self.left_backward.set_high()?;
        self.right_backward.set_high()?;
        self.left_forward.set_low()?;
        self.right_forward.set_low()?;
        Ok(())
    }

    /// Turn left
    fn turn_left(&mut self) -> anyhow::Result<()> {
        self.left_enable.set_high()?;
        self.right_enable.set_high()?;
        self.left_forward.set_low()?;
        self.right_forward.set_high()?;
        self.right_backward.set_low()?;
        self.left_backward.set_high()?;
        Ok(())
    }

    /// Turn right
    fn turn_right(&mut self) -> anyhow::Result<()> {
        self.left_enable.set_high()?;
        self.right_enable.set_high()?;
        self.left_forward.set_high()?;
        self.right_forward.set_low()?;
        self.right_backward.set_high()?;
        self.left_backward.set_low()?;
        Ok(())
    }

    /// Stop all motors
    fn stop(&mut self) -> anyhow::Result<()> {
        self.left_enable.set_low()?;
        self.right_enable.set_low()?;
        self.left_forward.set_low()?;
        self.left_backward.set_low()?;
        self.right_forward.set_low()?;
        self.right_backward.set_low()?;
        Ok(())
    }
}

fn setup_wifi(
    wifi: &mut BlockingWifi<EspWifi<'static>>,
) -> anyhow::Result<()> {
    let wifi_configuration = Configuration::Client(ClientConfiguration {
        ssid: WIFI_SSID.try_into().unwrap(),
        bssid: None,
        auth_method: AuthMethod::WPA2Personal,
        password: WIFI_PASSWORD.try_into().unwrap(),
        channel: None,
        ..Default::default()
    });

    wifi.set_configuration(&wifi_configuration)?;
    
    info!("Starting WiFi...");
    wifi.start()?;
    
    info!("Connecting to WiFi SSID: {}", WIFI_SSID);
    wifi.connect()?;
    
    info!("Waiting for DHCP lease...");
    wifi.wait_netif_up()?;
    
    let ip_info = wifi.wifi().sta_netif().get_ip_info()?;
    info!("WiFi connected! IP: {}", ip_info.ip);
    
    Ok(())
}

fn handle_mental_imagery(
    msg: &OscMessage,
    motors: &Arc<Mutex<MotorPins>>,
) -> anyhow::Result<()> {
    if msg.args.len() >= 2 {
        let right_prediction = match &msg.args[0] {
            OscType::Float(f) => *f,
            _ => return Ok(()),
        };
        
        let left_prediction = match &msg.args[1] {
            OscType::Float(f) => *f,
            _ => return Ok(()),
        };
        
        info!("Right: {}, Left: {}", right_prediction, left_prediction);
        
        let mut motors = motors.lock().unwrap();
        
        if left_prediction > 0.6 {
            motors.turn_left()?;
        } else if right_prediction > 0.6 {
            motors.turn_right()?;
        } else {
            motors.stop()?;
        }
    }
    
    Ok(())
}

fn osc_listener_thread(
    motors: Arc<Mutex<MotorPins>>,
    port: u16,
) -> anyhow::Result<()> {
    let socket = UdpSocket::bind(format!("0.0.0.0:{}", port))?;
    info!("UDP listener started on port {}", port);
    
    let mut buf = [0u8; rosc::decoder::MTU];
    let mut last_message_time = Instant::now();
    
    loop {
        // Set a timeout so we can check for motor timeout
        socket.set_read_timeout(Some(Duration::from_millis(100)))?;
        
        match socket.recv_from(&mut buf) {
            Ok((size, _addr)) => {
                let packet = rosc::decoder::decode_udp(&buf[..size]);
                
                match packet {
                    Ok((_, OscPacket::Message(msg))) => {
                        if msg.addr == "/neuropype" {
                            if let Err(e) = handle_mental_imagery(&msg, &motors) {
                                error!("Error handling message: {:?}", e);
                            }
                            last_message_time = Instant::now();
                        }
                    }
                    Ok((_, OscPacket::Bundle(bundle))) => {
                        for packet in bundle.content {
                            if let OscPacket::Message(msg) = packet {
                                if msg.addr == "/neuropype" {
                                    if let Err(e) = handle_mental_imagery(&msg, &motors) {
                                        error!("Error handling message: {:?}", e);
                                    }
                                    last_message_time = Instant::now();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error decoding OSC packet: {:?}", e);
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout - check if we should stop motors
                if last_message_time.elapsed() > Duration::from_millis(TIMEOUT_MILLIS) {
                    if let Ok(mut motors) = motors.lock() {
                        let _ = motors.stop();
                    }
                }
            }
            Err(e) => {
                error!("Error receiving UDP packet: {:?}", e);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    esp_idf_sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    
    info!("ESP Booted.");
    
    let mut peripherals = Peripherals::take()?;
    let sys_loop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;
    
    // Initialize motor pins
    let motors = Arc::new(Mutex::new(MotorPins::new(&mut peripherals)?));
    
    // Setup WiFi
    let mut wifi = BlockingWifi::wrap(
        EspWifi::new(peripherals.modem, sys_loop.clone(), Some(nvs))?,
        sys_loop,
    )?;
    
    setup_wifi(&mut wifi)?;
    
    // Start OSC listener in a separate thread
    let motors_clone = Arc::clone(&motors);
    thread::Builder::new()
        .stack_size(8192)
        .spawn(move || {
            if let Err(e) = osc_listener_thread(motors_clone, LISTEN_PORT) {
                error!("OSC listener thread error: {:?}", e);
            }
        })?;
    
    info!("System initialized. Listening for OSC messages on port {}", LISTEN_PORT);
    
    // Main loop - keep the program running
    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

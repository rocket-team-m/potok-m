//! UART transport and the 50 Hz service loop.
//!
//! The transport is a trait so the loop is testable without serial hardware.
//! The frame layout is a separate trait (`FrameLayout`) so the local protocol
//! copy can be swapped for the generated `potok-protocol` crate in one place.

use crate::frame::{FrameError, FrameLayout, ReplyFrame, RequestFrame, REPLY_LEN, REQUEST_LEN};
use crate::runtime::Runtime;
use crate::speculative::SpeculativeRuntime;
use serialport::SerialPort;
use std::io::Read;
use std::time::{Duration, Instant};

pub const LOOP_PERIOD: Duration = Duration::from_millis(20); // 50 Hz
pub const READ_TIMEOUT: Duration = Duration::from_millis(8);
pub const DEFAULT_BAUD: u32 = 921_600;

pub trait Transport {
    /// Read up to `buf.len()` bytes. Returns 0 on timeout.
    fn read_into(&mut self, buf: &mut [u8], timeout: Duration) -> Result<usize, String>;
    fn write_all(&mut self, buf: &[u8]) -> Result<(), String>;
}

pub struct SerialTransport {
    port: Box<dyn SerialPort>,
}

impl SerialTransport {
    pub fn open(device: &str, baud: u32) -> Result<Self, String> {
        let port = serialport::new(device, baud)
            .timeout(READ_TIMEOUT)
            .data_bits(serialport::DataBits::Eight)
            .parity(serialport::Parity::None)
            .stop_bits(serialport::StopBits::One)
            .flow_control(serialport::FlowControl::None)
            .open()
            .map_err(|e| format!("cannot open {device}: {e}"))?;
        Ok(Self { port })
    }
}

impl Transport for SerialTransport {
    fn read_into(&mut self, buf: &mut [u8], timeout: Duration) -> Result<usize, String> {
        self.port.set_timeout(timeout).map_err(|e| e.to_string())?;
        self.port.read(buf).map_err(|e| e.to_string())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), String> {
        self.port.write_all(buf).map_err(|e| e.to_string())
    }
}

/// Handle one decoded request and encode the reply. Kept separate from I/O
/// so it can be exercised with a mock transport.
pub fn handle_request<L: FrameLayout>(
    layout: &L,
    runtime: &mut Runtime,
    bytes: &[u8],
    reply_buf: &mut [u8; REPLY_LEN],
) -> Result<(), ()> {
    let req: RequestFrame = match layout.decode_request(bytes) {
        Ok(r) => r,
        Err(FrameError::VersionMismatch(_)) => {
            runtime.disable_now();
            return Err(());
        }
        Err(_) => {
            runtime.on_invalid_frame();
            return Err(());
        }
    };
    let reply = runtime.on_request(&req);
    layout.encode_reply(&reply, reply_buf).map_err(|_| ())
}

/// Speculative counterpart of `handle_request`. A bad frame gets a fault
/// reply (`ai_healthy = 0`) and never reaches either model.
pub fn handle_spec_request<L: FrameLayout>(
    layout: &L,
    runtime: &mut SpeculativeRuntime,
    bytes: &[u8],
    reply_buf: &mut [u8; REPLY_LEN],
) -> Result<(), ()> {
    let reply = match layout.decode_request(bytes) {
        Ok(req) => runtime.on_request(&req),
        Err(FrameError::VersionMismatch(_)) => {
            runtime.disable_now();
            ReplyFrame::muted(0, false)
        }
        Err(_) => {
            runtime.on_invalid_frame();
            ReplyFrame::muted(0, false)
        }
    };
    layout.encode_reply(&reply, reply_buf).map_err(|_| ())
}

/// Run until the process is stopped. One read per 20 ms tick, 8 ms read
/// timeout; invalid frames and timeouts count as misses (three disable AI).
pub fn run_loop<L: FrameLayout, T: Transport>(
    layout: L,
    transport: &mut T,
    mut runtime: Runtime,
) -> Result<(), String> {
    let mut read_buf = [0u8; REQUEST_LEN * 4];
    let mut reply_buf = [0u8; REPLY_LEN];
    let mut pending: Vec<u8> = Vec::with_capacity(read_buf.len());
    let mut was_disabled = false;
    eprintln!("potok-edge: inference loop started (50 Hz, 8 ms read timeout)");

    loop {
        let tick = Instant::now();
        match transport.read_into(&mut read_buf, READ_TIMEOUT) {
            Ok(0) => runtime.on_timeout(),
            Ok(n) => pending.extend_from_slice(&read_buf[..n]),
            Err(e) => {
                runtime.on_invalid_frame();
                eprintln!("potok-edge: UART read error: {e}");
            }
        }
        if pending.len() > read_buf.len() {
            // Desynchronized stream: keep the most recent bytes only.
            pending.drain(..pending.len() - read_buf.len());
        }
        while pending.len() >= REQUEST_LEN {
            let frame: Vec<u8> = pending.drain(..REQUEST_LEN).collect();
            match handle_request(&layout, &mut runtime, &frame, &mut reply_buf) {
                Ok(()) => {
                    if transport.write_all(&reply_buf).is_err() {
                        runtime.on_invalid_frame();
                    }
                }
                Err(()) => { /* counted inside handle_request */ }
            }
        }
        let elapsed = tick.elapsed();
        if elapsed < LOOP_PERIOD {
            std::thread::sleep(LOOP_PERIOD - elapsed);
        }
        if runtime.is_disabled() && !was_disabled {
            eprintln!("potok-edge: assistant disabled after 3 consecutive misses; AI stays off until restart");
            was_disabled = true;
        }
    }
}

/// Convenience entry point used by the `infer` binary.
pub fn serve(runtime: Runtime, device: &str, baud: u32) -> Result<(), String> {
    let mut transport = SerialTransport::open(device, baud)?;
    run_loop(crate::frame::PotokUartV1, &mut transport, runtime)
}

//! Throughput benchmark: episodes/s and frames/s, with and without sensors.
//! Run: cargo bench -p potok-sim --bench throughput

use std::time::Instant;

use potok_sim::{Config, Simulator};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    for (sensors, fast) in [(true, true), (false, true), (true, false)] {
        let mut cfg = Config::default();
        cfg.sim.sensors = sensors;
        cfg.run.fast_field = fast;
        cfg.run.seed = 1337;
        let sim = Simulator::new(cfg).unwrap();
        sim.run_episode(0).unwrap(); // warm-up
        let t0 = Instant::now();
        let mut frames = 0usize;
        for i in 0..n {
            frames += sim.run_episode(i as u64).unwrap().frames_generated;
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "sensors={sensors:<5} fast_field={fast:<5} episodes/s={:8.2} frames/s={:10.0}  ({n} episodes, {frames} frames, {secs:.3}s)",
            n as f64 / secs, frames as f64 / secs,
        );
    }
}

//! IMU Allan-deviation CSV -> validated [`ImuAllanPoints`].

use potok_sim::identification::{AllanPoint, ImuAllanPoints, ImuChannel};

use crate::csv;
use crate::error::{MeasurementError, Result};
use crate::provenance::SourceRecord;

pub const KIND: &str = "imu_allan";
pub const COLUMNS: &[&str] = &["tau_s", "sigma"];

/// One channel's Allan series together with the identity of its source file.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuAllanMeasurement {
    pub points: ImuAllanPoints,
    pub provenance: SourceRecord,
}

/// Parse and validate one Allan CSV. The channel is carried in metadata so a
/// gyro file can never be silently ingested as an accelerometer file.
pub fn ingest(text: &str, path: &str) -> Result<ImuAllanMeasurement> {
    let doc = csv::parse(text, path, COLUMNS)?;
    let declared = doc.meta_required("measurement")?;
    if declared != KIND {
        return Err(MeasurementError::Csv {
            path: path.to_string(),
            detail: format!(
                "metadata 'measurement' is '{declared}', expected '{KIND}'; point this recipe at the right file"
            ),
        });
    }
    let label = doc.meta_required("source")?.to_string();
    let channel = match doc.meta_required("channel")? {
        "gyro" => ImuChannel::Gyro,
        "accel" => ImuChannel::Accel,
        other => {
            return Err(MeasurementError::Csv {
                path: path.to_string(),
                detail: format!(
                    "metadata 'channel' value '{other}' is neither 'gyro' nor 'accel'; label the channel explicitly"
                ),
            })
        }
    };
    let points: Vec<AllanPoint> = doc
        .rows
        .iter()
        .map(|r| AllanPoint {
            tau_s: r.cells[0],
            sigma: r.cells[1],
        })
        .collect();
    let points = ImuAllanPoints::new(channel, points)
        .map_err(|e| MeasurementError::Identification(e.to_string()))?;
    let provenance = SourceRecord::new(&doc, text, KIND, &label);
    Ok(ImuAllanMeasurement { points, provenance })
}

//! Local date and time for display. Windows converts with the time-zone
//! rules in force at that date (summer time included); other platforms show UTC.

use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Local {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Local time of `ms` (milliseconds since 1970, UTC).
pub fn local(ms: u64) -> Local {
    #[cfg(target_os = "windows")]
    {
        if let Some(l) = windows_local(ms) {
            return l;
        }
    }
    utc(ms)
}

fn utc(ms: u64) -> Local {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    Local { year, month, day, hour: (rem / 3600) as u32, minute: (rem % 3600 / 60) as u32, second: (rem % 60) as u32 }
}

#[cfg(target_os = "windows")]
fn windows_local(ms: u64) -> Option<Local> {
    use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
    use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};
    // FILETIME: 100 ns ticks since 1601-01-01.
    let ticks = (ms + 11_644_473_600_000) * 10_000;
    let ft = FILETIME { dwLowDateTime: ticks as u32, dwHighDateTime: (ticks >> 32) as u32 };
    unsafe {
        let mut utc: SYSTEMTIME = std::mem::zeroed();
        let mut loc: SYSTEMTIME = std::mem::zeroed();
        if FileTimeToSystemTime(&ft, &mut utc) == 0 || SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut loc) == 0 {
            return None;
        }
        Some(Local {
            year: loc.wYear as i32,
            month: loc.wMonth as u32,
            day: loc.wDay as u32,
            hour: loc.wHour as u32,
            minute: loc.wMinute as u32,
            second: loc.wSecond as u32,
        })
    }
}

/// Days since 1970-01-01 -> (year, month, day). Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }) as i32, m, d)
}

/// (year, month, day) -> days since 1970-01-01.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

impl Local {
    fn day_number(&self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
    }
    fn weekday(&self) -> &'static str {
        // 1970-01-01 was a Thursday (index 3 with Monday = 0).
        WEEKDAYS[(self.day_number() + 3).rem_euclid(7) as usize]
    }
}

/// "Today 18:42", "Yesterday 09:10", "Mon 14:05", "3 Nov 14:05", "3 Nov 2025".
pub fn when(ms: u64, now: u64) -> String {
    let t = local(ms);
    let n = local(now);
    let days = n.day_number() - t.day_number();
    let hm = format!("{:02}:{:02}", t.hour, t.minute);
    match days {
        0 => format!("Today {}", hm),
        1 => format!("Yesterday {}", hm),
        2..=6 => format!("{} {}", t.weekday(), hm),
        _ if t.year == n.year => format!("{} {} {}", t.day, MONTHS[(t.month - 1) as usize], hm),
        _ => format!("{} {} {}", t.day, MONTHS[(t.month - 1) as usize], t.year),
    }
}

/// "18:42:07" local.
pub fn clock(ms: u64) -> String {
    let t = local(ms);
    format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second)
}

/// Parse an RFC 3339 UTC timestamp as written by the log ("2026-11-08T18:04:50.162151Z").
pub fn parse_utc(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[10] != b'T' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<u32>().ok();
    let (y, mo, d) = (num(0..4)? as i32, num(5..7)?, num(8..10)?);
    let (h, mi, se) = (num(11..13)? as u64, num(14..16)? as u64, num(17..19)? as u64);
    let frac_ms = s.get(20..23).and_then(|f| f.parse::<u64>().ok()).unwrap_or(0);
    let days = days_from_civil(y, mo, d);
    if days < 0 {
        return None;
    }
    Some(((days as u64 * 86_400) + h * 3600 + mi * 60 + se) * 1000 + frac_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_round_trip() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for d in [-1000, 0, 365, 20_000, 20_765] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        let t = utc(parse_utc("2026-11-08T18:04:50.162151Z").unwrap());
        assert_eq!((t.year, t.month, t.day, t.hour, t.minute, t.second), (2026, 11, 8, 18, 4, 50));
        assert_eq!(utc(0).weekday(), "Thu");
    }
}

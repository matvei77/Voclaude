//! "Start with Windows": the per-user Run key value `Voclaude`, the same value
//! the installer writes on the first install. Other platforms: not supported.

#[cfg(target_os = "windows")]
const RUN_KEY: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
#[cfg(target_os = "windows")]
const VALUE: &str = "Voclaude";

#[cfg(target_os = "windows")]
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn supported() -> bool {
    cfg!(target_os = "windows")
}

/// The command Windows runs at sign-in, if the value is set.
pub fn current() -> Option<String> {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_SZ};
        let (key, value) = (wide(RUN_KEY), wide(VALUE));
        let mut size: u32 = 0;
        if RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), std::ptr::null_mut(), &mut size) != 0 {
            return None;
        }
        let mut buf = vec![0u16; (size as usize).div_ceil(2)];
        if RegGetValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), buf.as_mut_ptr().cast(), &mut size) != 0 {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..len]))
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// The command for this copy of Voclaude: the quoted path of the running exe.
pub fn this_command() -> Option<String> {
    std::env::current_exe().ok().map(|p| format!("\"{}\"", p.display()))
}

/// True when the Run value starts this copy (compared without quotes and case).
pub fn points_here(command: &str) -> bool {
    let norm = |s: &str| s.trim().trim_matches('"').to_lowercase();
    this_command().is_some_and(|c| norm(&c) == norm(command))
}

/// Start this copy at sign-in (`true`), or remove the value (`false`).
pub fn set(enabled: bool) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Registry::{RegDeleteKeyValueW, RegSetKeyValueW, HKEY_CURRENT_USER, REG_SZ};
        let (key, value) = (wide(RUN_KEY), wide(VALUE));
        let status = if enabled {
            let command = wide(&this_command().ok_or("cannot find this program's path")?);
            RegSetKeyValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr(), REG_SZ, command.as_ptr().cast(), (command.len() * 2) as u32)
        } else {
            match RegDeleteKeyValueW(HKEY_CURRENT_USER, key.as_ptr(), value.as_ptr()) {
                2 => 0, // ERROR_FILE_NOT_FOUND: already off
                s => s,
            }
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format!("registry error {}", status))
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = enabled;
        Err("not supported on this platform".into())
    }
}

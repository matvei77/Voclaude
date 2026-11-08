//! One tray app per profile (data folder).
//!
//! The first `voclaude` holds a named mutex. A second launch (Start menu,
//! desktop icon, `voclaude console` from a terminal) does not start another
//! tray app: it signals a named event and exits, and the running app opens
//! its console window. Names include a hash of the data folder, so a test
//! profile (`VOCLAUDE_HOME`) runs beside the normal one. Other platforms: no-op.

use crossbeam_channel::Sender;

/// Held for the life of the tray app.
pub struct Instance {
    #[cfg(target_os = "windows")]
    _mutex: isize,
}

fn suffix() -> String {
    let dir = crate::config::project_dirs()
        .map(|d| d.data_dir().to_string_lossy().to_lowercase())
        .unwrap_or_default();
    // FNV-1a: stable across runs and builds (std's hasher is not).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in dir.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", h)
}

#[cfg(target_os = "windows")]
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Take the profile's instance lock. `None`: another tray app holds it.
#[cfg(target_os = "windows")]
pub fn acquire() -> Option<Instance> {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS};
    use windows_sys::Win32::System::Threading::CreateMutexW;
    let name = wide(&format!("Local\\Voclaude-{}", suffix()));
    unsafe {
        let handle = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
        if handle.is_null() {
            return Some(Instance { _mutex: 0 }); // cannot tell; behave as before
        }
        if GetLastError() == ERROR_ALREADY_EXISTS {
            CloseHandle(handle);
            return None;
        }
        Some(Instance { _mutex: handle as isize })
    }
}

#[cfg(not(target_os = "windows"))]
pub fn acquire() -> Option<Instance> {
    Some(Instance {})
}

/// True when a tray app runs for this profile.
pub fn is_running() -> bool {
    #[cfg(target_os = "windows")]
    {
        match acquire() {
            None => true,
            Some(instance) => {
                drop(instance);
                false
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

#[cfg(target_os = "windows")]
impl Drop for Instance {
    fn drop(&mut self) {
        if self._mutex != 0 {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self._mutex as _) };
        }
    }
}

/// Ask the running tray app to open (or bring forward) its console.
/// Returns false if no tray app listens.
pub fn signal_open() -> bool {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};
        use windows_sys::Win32::UI::WindowsAndMessaging::{AllowSetForegroundWindow, ASFW_ANY};
        let name = wide(&format!("Local\\Voclaude-open-{}", suffix()));
        let handle = OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr());
        if handle.is_null() {
            return false;
        }
        // This process was started by the user, so it may hand the
        // foreground to the window the tray app is about to show.
        AllowSetForegroundWindow(ASFW_ANY);
        let ok = SetEvent(handle) != 0;
        CloseHandle(handle);
        ok
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// In the tray app: send `event` on `tx` each time another launch signals.
pub fn listen_for_open<E: Clone + Send + 'static>(tx: Sender<E>, event: E) {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};
        let name = wide(&format!("Local\\Voclaude-open-{}", suffix()));
        let handle = CreateEventW(std::ptr::null(), 0, 0, name.as_ptr());
        if handle.is_null() {
            tracing::warn!("Could not create the open-console event");
            return;
        }
        let handle = handle as isize;
        let spawned = std::thread::Builder::new().name("open-console".into()).spawn(move || loop {
            if WaitForSingleObject(handle as _, INFINITE) != 0 {
                break;
            }
            if tx.send(event.clone()).is_err() {
                break;
            }
        });
        if let Err(e) = spawned {
            tracing::warn!("Could not start the open-console listener: {}", e);
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (tx, event);
    }
}

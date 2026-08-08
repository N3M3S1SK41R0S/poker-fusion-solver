//! Sonde de faisabilité Phase 1 — le go/no-go du Plan Directeur §2.2.
//!
//! Question posée : peut-on capturer une fenêtre de client poker (même
//! **occultée**) avec l'API Windows Graphics Capture (crate
//! `windows-capture` 2.0), avec un coût de lecture par région d'intérêt
//! p95 < 1 ms ?
//!
//! Mesures par frame, sur N frames :
//!   - `copy_us`  : mappage GPU→CPU du buffer COMPLET + parcours (µs) —
//!     borne haute, rapportée à titre informatif ;
//!   - `crop_us`  : readback d'une ROI 200×100 via `buffer_crop` — le
//!     coût RÉEL du scraper (cartes, stacks, boutons) ; c'est le critère ;
//!   - `inter_us` : intervalle entre frames (µs) — WGC n'émet que sur
//!     mise à jour de la fenêtre ;
//!   - `mean_px`  : moyenne des octets échantillonnés — un client qui se
//!     protégerait par WDA_EXCLUDEFROMCAPTURE rendrait du noir (≈ 0).
//!
//! Sortie : une ligne JSON sur stdout. Codes de sortie : 0 ok, 2 fenêtre
//! introuvable, 3 échec de capture, 4 timeout (fenêtre trouvée mais
//! aucune/pas assez de frames — fenêtre statique ou capture bloquée).
//!
//! Usage :
//!   probe [sous-chaîne du titre] [nb frames] [--snap chemin.png] [--timeout s]
//!   probe --auto [nb frames] …     (détecte un client poker connu)
//!   probe --list                   (énumère les fenêtres capturables)

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::encoder::ImageFormat;
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

/// Rooms reconnues par --auto (sous-chaînes de titre, insensible à la casse).
/// La couche capture est agnostique de la room : seule la DÉTECTION est
/// spécifique ; la lecture du contenu (pfs-vision) aura ses gabarits par room.
const ROOMS: &[&str] = &[
    "pmu",
    "winamax",
    "pokerstars",
    "partypoker",
    "party poker",
    "unibet",
    "betclic",
    "888poker",
    "ggpoker",
    "ipoker",
    "pokerclient",
    "no limit hold",
    "pot limit omaha",
];

/// Fenêtres à ne JAMAIS prendre pour un client (notre propre outillage).
const EXCLUDE: &[&str] = &["fusion", "pfs-", "probe"];

static FINISHED: AtomicBool = AtomicBool::new(false);

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (idx - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn stats(mut v: Vec<f64>) -> (f64, f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile(&v, 50.0),
        percentile(&v, 95.0),
        percentile(&v, 99.0),
        v.last().copied().unwrap_or(f64::NAN),
    )
}

struct ProbeFlags {
    target: usize,
    snap: Option<String>,
}

struct Probe {
    copy_us: Vec<f64>,
    crop_us: Vec<f64>,
    inter_us: Vec<f64>,
    px_means: Vec<f64>,
    last_arrival: Option<Instant>,
    flags: ProbeFlags,
    snapped: bool,
    width: u32,
    height: u32,
}

impl GraphicsCaptureApiHandler for Probe {
    type Flags = ProbeFlags;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        let n = ctx.flags.target;
        Ok(Self {
            copy_us: Vec::with_capacity(n),
            crop_us: Vec::with_capacity(n),
            inter_us: Vec::with_capacity(n),
            px_means: Vec::with_capacity(n),
            last_arrival: None,
            flags: ctx.flags,
            snapped: false,
            width: 0,
            height: 0,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let now = Instant::now();
        if let Some(prev) = self.last_arrival {
            self.inter_us
                .push(now.duration_since(prev).as_secs_f64() * 1e6);
        }
        self.last_arrival = Some(now);

        // preuve visuelle : première frame → PNG
        if !self.snapped {
            if let Some(path) = self.flags.snap.clone() {
                frame.save_as_image(&path, ImageFormat::Png)?;
                eprintln!("snapshot : {path}");
            }
            self.snapped = true;
        }

        // borne haute : buffer complet, lectures échantillonnées 1/1024
        let t0 = Instant::now();
        let mut buffer = frame.buffer()?;
        let raw = buffer.as_raw_buffer();
        let mut acc: u64 = 0;
        let mut n: u64 = 0;
        for &b in raw.iter().step_by(1024) {
            acc = acc.wrapping_add(u64::from(b));
            n += 1;
        }
        std::hint::black_box(acc);
        self.copy_us.push(t0.elapsed().as_secs_f64() * 1e6);
        if n > 0 {
            self.px_means.push(acc as f64 / n as f64);
        }
        self.width = frame.width();
        self.height = frame.height();

        // le coût réel du scraper : ROI 200×100 au centre
        let (w, h) = (self.width, self.height);
        if w > 300 && h > 200 {
            let (x, y) = (w / 2 - 100, h / 2 - 50);
            let t1 = Instant::now();
            let mut roi = frame.buffer_crop(x, y, x + 200, y + 100)?;
            std::hint::black_box(roi.as_raw_buffer().first().copied());
            self.crop_us.push(t1.elapsed().as_secs_f64() * 1e6);
        }

        if self.copy_us.len() >= self.flags.target {
            let (c50, c95, c99, cmax) = stats(self.copy_us.clone());
            let (r50, r95, r99, rmax) = stats(self.crop_us.clone());
            let (i50, i95, i99, imax) = stats(self.inter_us.clone());
            let mean_px =
                self.px_means.iter().sum::<f64>() / self.px_means.len().max(1) as f64;
            println!(
                "{{\"frames\":{},\"width\":{},\"height\":{},\
                 \"copy_us\":{{\"p50\":{c50:.1},\"p95\":{c95:.1},\"p99\":{c99:.1},\"max\":{cmax:.1}}},\
                 \"crop_us\":{{\"p50\":{r50:.1},\"p95\":{r95:.1},\"p99\":{r99:.1},\"max\":{rmax:.1}}},\
                 \"inter_us\":{{\"p50\":{i50:.1},\"p95\":{i95:.1},\"p99\":{i99:.1},\"max\":{imax:.1}}},\
                 \"mean_px\":{mean_px:.1},\
                 \"go\":{}}}",
                self.copy_us.len(),
                self.width,
                self.height,
                r95 < 1000.0,
            );
            FINISHED.store(true, Ordering::SeqCst);
            capture_control.stop();
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        eprintln!("fenêtre fermée avant la fin de la mesure");
        Ok(())
    }
}

/// Cherche une fenêtre de client poker connue (hors notre outillage).
fn find_room_window() -> Option<(Window, String)> {
    let windows = Window::enumerate().ok()?;
    for w in windows {
        let Ok(title) = w.title() else { continue };
        let lower = title.to_lowercase();
        if EXCLUDE.iter().any(|e| lower.contains(e)) {
            continue;
        }
        if ROOMS.iter().any(|r| lower.contains(r)) {
            return Some((w, title));
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list") {
        match Window::enumerate() {
            Ok(windows) => {
                for w in windows {
                    if let Ok(title) = w.title() {
                        if !title.trim().is_empty() {
                            println!("{title}");
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("énumération impossible : {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // options nommées
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let snap = opt("--snap");
    let timeout_s: u64 = opt("--timeout").and_then(|s| s.parse().ok()).unwrap_or(30);
    let auto = args.iter().any(|a| a == "--auto");

    // positionnels (hors options et leurs valeurs)
    let mut positional: Vec<String> = Vec::new();
    let mut skip = false;
    for a in args.iter().skip(1) {
        if skip {
            skip = false;
            continue;
        }
        if a == "--snap" || a == "--timeout" {
            skip = true;
            continue;
        }
        if a.starts_with("--") {
            continue;
        }
        positional.push(a.clone());
    }
    let frames: usize = positional
        .iter()
        .filter_map(|s| s.parse().ok())
        .next()
        .unwrap_or(300);

    let window = if auto {
        match find_room_window() {
            Some((w, title)) => {
                eprintln!("room détectée : {title}");
                w
            }
            None => {
                eprintln!("aucun client poker connu à l'écran (rooms : {ROOMS:?})");
                std::process::exit(2);
            }
        }
    } else {
        let needle = positional
            .iter()
            .find(|s| s.parse::<usize>().is_err())
            .cloned()
            .unwrap_or_else(|| "Winamax".to_string());
        match Window::from_contains_name(&needle) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("aucune fenêtre dont le titre contient « {needle} » : {e}");
                eprintln!("fenêtres disponibles : probe --list");
                std::process::exit(2);
            }
        }
    };
    if let Ok(title) = window.title() {
        eprintln!("capture de : {title} ({frames} frames, timeout {timeout_s}s)");
    }

    // garde-fou : une fenêtre statique (lobby figé) ou une capture bloquée
    // (WDA_EXCLUDEFROMCAPTURE) ne produirait jamais assez de frames.
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(timeout_s));
        if !FINISHED.load(Ordering::SeqCst) {
            eprintln!(
                "timeout {timeout_s}s — pas assez de frames (fenêtre statique, \
                 minimisée, ou capture bloquée par le client)"
            );
            std::process::exit(4);
        }
    });

    let settings = Settings::new(
        window,
        CursorCaptureSettings::WithoutCursor,
        DrawBorderSettings::WithoutBorder,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Bgra8,
        ProbeFlags {
            target: frames,
            snap,
        },
    );

    if let Err(e) = Probe::start(settings) {
        eprintln!("échec de la capture : {e}");
        std::process::exit(3);
    }
}

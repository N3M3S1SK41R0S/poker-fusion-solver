//! Sonde de faisabilité Phase 1 — le go/no-go du Plan Directeur §2.2.
//!
//! Question posée : peut-on capturer une fenêtre Winamax **occultée** avec
//! l'API Windows Graphics Capture (crate `windows-capture` 2.0), avec un
//! coût de lecture par frame p95 < 1 ms ?
//!
//! Ce binaire capture la fenêtre dont le titre contient la sous-chaîne
//! donnée, mesure sur N frames :
//!   - `copy_us`  : temps de mappage GPU→CPU + parcours du buffer (µs) —
//!     c'est le coût que la perception ajoute à chaque frame ;
//!   - `inter_us` : intervalle entre frames consécutives (µs) — borné par
//!     le rythme de rafraîchissement WGC (~60 Hz par défaut) ;
//!   - `mean_px`  : moyenne des octets échantillonnés — une fenêtre occultée
//!     dont la capture échouerait donnerait un buffer noir (mean ≈ 0) ;
//!     c'est la preuve que WGC lit bien le contenu, pas l'écran.
//!
//! Sortie : une ligne JSON sur stdout (consommée par le harnais de test).
//!
//! Usage : probe [sous-chaîne du titre] [nb frames]
//!         probe --list        (énumère les fenêtres capturables)

use std::time::Instant;

use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

/// Percentile par interpolation linéaire sur un tableau trié.
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

struct Probe {
    copy_us: Vec<f64>,
    crop_us: Vec<f64>,
    inter_us: Vec<f64>,
    px_means: Vec<f64>,
    last_arrival: Option<Instant>,
    target: usize,
    width: u32,
    height: u32,
}

impl GraphicsCaptureApiHandler for Probe {
    type Flags = usize;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            copy_us: Vec::with_capacity(ctx.flags),
            crop_us: Vec::with_capacity(ctx.flags),
            inter_us: Vec::with_capacity(ctx.flags),
            px_means: Vec::with_capacity(ctx.flags),
            last_arrival: None,
            target: ctx.flags,
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

        // Le coût réel de la perception : mapper la texture et LIRE les octets.
        // Échantillonnage 1/1024 pour forcer des lectures répandues sur tout
        // le buffer sans payer un memcpy complet (le scraper réel ne lira que
        // des régions d'intérêt).
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

        // Le coût RÉEL du scraper : readback d'une région d'intérêt
        // (cartes, stack, bouton ≈ 200×100 px), pas du buffer complet.
        // buffer_crop fait un CopySubresourceRegion GPU → staging réduit.
        let (w, h) = (self.width, self.height);
        if w > 300 && h > 200 {
            let (x, y) = (w / 2 - 100, h / 2 - 50);
            let t1 = Instant::now();
            let mut roi = frame.buffer_crop(x, y, x + 200, y + 100)?;
            std::hint::black_box(roi.as_raw_buffer().first().copied());
            self.crop_us.push(t1.elapsed().as_secs_f64() * 1e6);
        }

        if self.copy_us.len() >= self.target {
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
            capture_control.stop();
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        eprintln!("fenêtre fermée avant la fin de la mesure");
        Ok(())
    }
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

    let needle = args.get(1).cloned().unwrap_or_else(|| "Winamax".to_string());
    let frames: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let window = match Window::from_contains_name(&needle) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("aucune fenêtre dont le titre contient « {needle} » : {e}");
            eprintln!("fenêtres disponibles : probe --list");
            std::process::exit(2);
        }
    };
    if let Ok(title) = window.title() {
        eprintln!("capture de : {title} ({frames} frames)");
    }

    let settings = Settings::new(
        window,
        CursorCaptureSettings::WithoutCursor,
        DrawBorderSettings::WithoutBorder,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Bgra8,
        frames,
    );

    if let Err(e) = Probe::start(settings) {
        eprintln!("échec de la capture : {e}");
        std::process::exit(3);
    }
}

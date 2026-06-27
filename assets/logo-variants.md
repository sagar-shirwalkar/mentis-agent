# Logo Variants — Meredith

## Primary: `meredith.svg`
Abstract isometric "M" with Meredith effect color story:
- **Left (cool intake)**: blue gradient (#4a8fe5 → #1a3a5c)
- **Center (radiator/heat exchange)**: amber gradient (#ffcc00 → #e67e22)
- **Right (thrust)**: red gradient (#ff4400 → #aa1100)
- Animated particles show flow direction; wordmark at bottom

## Favicon (`favicon.ico` / `favicon.svg`)
Crop to the isometric M only (no wordmark, no thrust trail), center at
(0, -20), viewBox="-120 -140 240 240". Remove animation for static
favicon. Can render from `concept3_heat_to_thrust.svg` with those
adjustments.

## Dark Mode (already dark background)
Background: #0a0a14 → #141428. Already optimized for dark themes —
the glow and hot colors pop against the dark background.

## Light Mode (`meredith-light.svg`)
Swap background to light (white or #f5f5f7), reduce element opacity
to ~60% of dark-mode values, soften the glow filters. Stroke colors
become darker (e.g. coolF stroke → #2a5a8c).

## Monochrome / Print (`meredith-mono.svg`)
Single-color version (black or #333) for print, docs, grayscale.
Remove gradients — use solid fills with opacity to show depth:
- Outer legs: 100% opacity
- Inner legs: 70% opacity
- Thrust trail: 40% opacity

## Small / App Icon (512×512)
Crop tightly around the M letterform. Remove intake lines, heat
diamonds, thrust trail, wordmark. Glow and radiator fin lines can
be kept but simplified. viewBox="-100 -100 200 160".

## Variant generation
Run `uv run scripts/generate_logo_concepts.py --variant <name>`
to generate a specific variant from the base template.

"""Render the UX mockups (design only, not shipped) and compose before/after PNGs into screenshots/ux/."""
import os
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "..", "..", "screenshots", "ux")
BEFORE = {1: "screenshots/h/H1_brain_neck_3d.png", 2: "screenshots/h/H4_neck_refused_card_menu.png", 3: None,
          4: "screenshots/h/H1_brain_neck_3d.png", 5: "screenshots/ux/before_5_series.png", 6: "screenshots/h/H3_neck_ok_card.png"}
TITLE = {1: "1  Camera: pan, focus on a part, frame Head / Neck", 2: "2  One 'Add scan' button + a visible library",
         3: "3  The mouth: sealed, translucent skin + honest note", 4: "4  Left panel: rail + search + presets + inspector",
         5: "5  Series list in plain words", 6: "6  New name, plain status, first-run progress"}
HL = {2: [(1440, 20, 1600, 60), (1150, 20, 1230, 60), (950, 50, 1240, 200)], 6: [(0, 0, 330, 45)], 1: [(860, 580, 1000, 960)],
      4: [(0, 60, 310, 930)], 5: [(15, 260, 310, 620)]}
font = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    for i in range(1, 7):
        pg.goto(f"file://{D}/a{i}.html"); pg.wait_for_timeout(300)
        pg.screenshot(path=f"{D}/after{i}.png")
    b.close()
for i in range(1, 7):
    after = Image.open(f"{D}/after{i}.png").convert("RGB")
    if BEFORE[i] is None:
        W, H = 1280, 800 + 70; c = Image.new("RGB", (W, H), "#0a0e18"); c.paste(after, (0, 70))
    else:
        bf = Image.open(BEFORE[i]).convert("RGB"); dr = ImageDraw.Draw(bf)
        for box in HL.get(i, []): dr.rectangle(box, outline="#ff6b6b", width=5)
        bf = bf.resize((1280, 768)); W, H = 2600, 900; c = Image.new("RGB", (W, H), "#0a0e18")
        c.paste(bf, (10, 110)); c.paste(after, (1310, 94))
        d = ImageDraw.Draw(c); d.text((14, 70), "BEFORE (current build, public sample)", fill="#ff9c9c", font=font(24))
        d.text((1314, 60), "AFTER (mockup)", fill="#8fe3b5", font=font(24))
    ImageDraw.Draw(c).text((14, 16), TITLE[i], fill="#e8edf5", font=font(30))
    names = {1: "camera", 2: "add_scan_library", 3: "mouth", 4: "panel", 5: "series_picker", 6: "name_status"}
    c.save(f"{OUT}/ux{i}_{names[i]}_before_after.png"); print("saved", i)

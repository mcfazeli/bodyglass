# Installing BodyGlass on your Mac

For Macs with Apple Silicon (M1 or newer) on macOS 13 or later. You need about 4 GB free disk space (about 6 GB with the optional neck labelling)
and internet access **once**, for the setup. You do **not** need admin rights, Homebrew or Python: the
setup puts a private copy of everything inside the BodyGlass folder.

> **Your scans never leave your Mac.** All processing runs on this computer. The app is a local page
> at `http://127.0.0.1:8765`, which only your own Mac can open. Nothing is uploaded.
> BodyGlass is for education only. It is not a medical device and not for diagnosis.

## Install (new users, one time, about 8–15 minutes)
1. **Unzip** `BodyGlass-mac.zip` (double-click it). Move the `BodyGlass` folder somewhere permanent,
   e.g. your home folder or `Applications`.
2. **Open Terminal**: press ⌘-Space, type `Terminal`, press Return.
3. **Go into the folder**: type `cd ` (with a space after it), drag the `BodyGlass` folder from Finder
   onto the Terminal window, and press Return.
4. **Run the setup**: type `./install.sh` and press Return. It downloads and installs, all inside the folder:
   * `uv` (a small Python manager) and its own Python 3.12 (in `.tools/`)
   * Python packages including PyTorch with Apple-GPU (MPS) support (in `.venv/`, about 1.5 GB)
   * AI model weights, checked with SHA-256: SynthStrip (29 MB, brain outline) and FastSurfer v2.6.0
     (16 MB code + 67 MB weights, detailed structures)
   * optional: neck labelling (TotalSegmentator 2.18, about 1.7 GB: packages + 4 model files). Its usage
     statistics are switched off before it first runs, and it never goes online while you use the app. If
     this download fails, everything else still works; run `./install.sh` again later to retry.

   It ends with `Done.` and a line saying whether the Apple GPU is available. If something fails (for
   example the Wi-Fi drops), just run `./install.sh` again: finished steps are skipped.
5. **Start the app**: double-click `BodyGlass.command` in Finder. A Terminal window opens (keep it open)
   and your browser shows BodyGlass.

### If macOS says "unidentified developer" or "cannot be opened"
The app isn't signed by Apple. Either:
* in Finder, **right-click** (or Control-click) `BodyGlass.command` → **Open** → **Open**, once; or
* in Terminal, inside the folder: `xattr -dr com.apple.quarantine .` (this removes the download flag from
  this folder only).

If you get "permission denied" for `./install.sh`, run `chmod +x install.sh BodyGlass.command` and try again.

## Using it
1. **＋ Add scan** (top right) and choose a folder. A **whole export** from your clinic or your hospital's patient portal is fine:
   BodyGlass reads the DICOM headers and sorts it by itself. The sharp 3D T1 head scan becomes the 3D model, other
   head scans (FLAIR, T2, SWI, DWI, ADC) are shown on the slices, and a neck / cervical-spine MRI is added as the
   neck. Added a neck folder on its own? It joins the model that is open. **Scans → Which scans are used?** explains
   every series in plain words; **Use for 3D instead** builds a model from another suitable series.
2. The quick 3D model takes about a minute; the detailed structures (lobes, hippocampus, ventricles…) follow a few
   minutes later while you look around. A short banner says when something you started begins and when it is ready.
3. **Moving around:** drag to rotate; **two-finger swipe** (trackpad), right-drag or ⇧-drag to move; pinch or the
   mouse wheel to zoom. **Double-click** a part (or press **F**) to orbit around it; **Head / Neck / Head + neck**
   at the top frames a region; **0** fits everything, **R** resets. Prefer swipe-to-zoom? About → tick
   "Two-finger swipe zooms".
4. **Left rail:** **Explore** (search any part, ready-made views, show/hide parts), **Layers** (peel slider, layer
   opacity, structure list), **Scans** (which scans are used, other scan types), **About**. Click the open tab
   again to hide the panel. Clicking a part opens its card on the right with **Focus** and **Hide**.
5. **Slice bar** (bottom left): tick Axial / Coronal / Sagittal and drag their sliders; **Cut away** opens the 3D
   model at the slices; **⋯** has "Keep deep structures whole", "Colour structures on the slices" and contrast.
   Slice pictures (bottom right): drag up/down or scroll over one to move through the slices.
6. **⌘K** finds any command or body part. **?** lists all shortcuts. **▶ Tour** gives a guided tour (it also starts
   by itself the first time you open a model).
7. **My body ▾** (top left) is your library: open, close, rename or delete models, see each model's exams and
   dates, replace the neck exam, and **Show in Finder**. Models are **saved automatically** in
   `~/Library/Application Support/BodyGlass`. Delete removes the 3D model only, never your scan folder.
8. **Tools ▾:** **Measure** (M), **Export for 3D printing** (E, STL / 3MF, life size in mm) and **Compare with an
   example brain** (X, a public healthy volunteer; one person, not a normal range, **not a diagnosis**).
9. **Skin:** shown 25 % see-through by default. Teeth and air give no MRI signal, so the surface is smoothed across
   the mouth and nostrils. Models made before this smoothing offer it once ("Smooth the mouth area now", skin only).

## Ask AI (optional, off until you choose)
Ask questions about your scan in plain words ("what does this part do?", "what does my report mean?").
Educational only, not a diagnosis. Click **✦ Ask AI** (top right). The first time, you choose one of two ways:

- **🔒 Private, on this Mac:** click **Set up**. BodyGlass picks the model for your Mac's memory (Qwen3.5-4B, or
  Qwen3.5-9B with 24 GB or more; both read text *and* pictures), adds the AI engine and downloads the model once
  (about 9 GB; 3 to 6 GB kept). The panel shows progress and the time left; you can close it and keep exploring,
  the percentage shows on the Ask AI button. After that, answers never leave your Mac and work offline.
  Short on disk? The panel offers a smaller ready-made copy (3 GB) instead.
- **☁ Use my ChatGPT (Plus or Pro, no API key):** click **Sign in with ChatGPT**, sign in in your browser and allow
  BodyGlass to use your plan. The first time, BodyGlass asks once whether it may send your question to OpenAI.
  Your sign-in stays in the macOS Keychain; **Sign out** is in the model menu at the top of the panel.

**Asking:** type and press Return, or tap a suggested question. Answers appear as they are written; ■ stops them.
Part names in an answer are links: click one to see that part in 3D. **✦ Ask AI** on a part's card asks about it.

**What the AI sees** is shown as small chips above the message box: the selected part and its volume, a measurement,
what is on screen. Click × on a chip to leave it out. With ChatGPT, the line under the chips says it goes to
OpenAI, and **See exact text** shows exactly what is sent.

**Pictures:** 📎 adds the current slice, a snapshot of the 3D view, or a picture file (or drag one onto the panel).
A sagittal slice or a view with the skin on can show your face: the chip warns you. If the model you picked reads
only text, BodyGlass answers that message with your picture-reading model and says so.

**Your report:** 📎 › My report, or paste it into the message box, then **Explain in plain words**. The parts it
mentions light up in 3D.

**Models and chats:** the menu at the top of the panel switches between private models and ChatGPT
(🖼 = reads pictures). **Manage models…** has the medical-tuned MedPsy-4B (text only), 8-bit versions and Delete.
Models live in `~/Library/Application Support/BodyGlass/models`. Chats are saved on this Mac per person: 🕘 shows
them, ✎ starts a new one.

## If something goes wrong
* **Labelling fails or hangs on the Apple GPU:** quit, then start it on the CPU from Terminal in the folder:
  `BODYGLASS_DEVICE=cpu ./BodyGlass.command` (slower). The quick model works either way.
* Skip the detailed structures entirely: `BODYGLASS_PHASE2=0 ./BodyGlass.command`.
* Open a folder directly: `./BodyGlass.command /path/to/DICOM-folder`.
* Port already in use: `./BodyGlass.command --port 8777`.

## Updating to a newer BodyGlass later
1. Quit the app (close its Terminal window).
2. Unzip the new `BodyGlass-mac.zip` somewhere else, e.g. in Downloads.
3. Copy the new files **over** your installed folder. In Terminal:
   `rsync -a ~/Downloads/BodyGlass/ /path/to/your/BodyGlass/`
   (type `rsync -a `, drag the **new** folder in, type `/`, a space, drag your **installed** folder in, type `/`, press Return.)
   Do *not* use Finder's "Replace" on the whole folder: that would delete the installed environment.
4. In the installed folder run `./install.sh` (quick: finished steps are skipped; it also removes files older
   versions left behind), then start `BodyGlass.command` as usual.

## Quit
Close the Terminal window that opened with the app (or press Ctrl-C in it). Then close the browser tab.

## Uninstall
1. Quit the app. If you used ChatGPT sign-in, click **Sign out** in the chat panel first (or delete the "BodyGlass"
   item in Keychain Access).
2. Drag the `BodyGlass` folder to the Trash. That removes the program, Python, packages and models.
3. Delete your saved models: in Finder choose Go → Go to Folder…, enter
   `~/Library/Application Support/BodyGlass` and move that folder to the Trash.

Nothing else is installed anywhere on your Mac.

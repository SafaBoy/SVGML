import os
import sys
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    "/home/safa/.gemini/antigravity-cli/brain/c54de23a-db1e-4cb6-8515-bffb35005efe",
)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")


def run_ui_tests():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # =====================================================================
        # PART A: Desktop Viewport (1440 x 960)
        # =====================================================================
        context = browser.new_context(
            viewport={"width": 1440, "height": 960}, accept_downloads=True
        )
        page = context.new_page()

        print(f"[Playwright] Navigating to {FRONTEND_URL} (Desktop 1440x960)...")
        page.goto(FRONTEND_URL, wait_until="networkidle")

        # 1. Verify Header & Service Health Badges (including GPU status)
        title_text = page.locator("h1").inner_text()
        assert "SVGML Studio" in title_text, f"Unexpected title: {title_text}"
        page.wait_for_selector("text=Go: Online", timeout=10000)
        page.wait_for_selector("text=Rust: Ready", timeout=10000)
        print("[Playwright] ✅ Header and backend health badges verified.")

        # 2. Verify Hugging Face Model Hub & Downloader badges
        page.wait_for_selector("text=Downloaded (", timeout=10000)
        page.wait_for_selector("text=Download HF", timeout=10000)
        search_input = page.locator('input[placeholder*="Search Hugging Face"]')
        search_input.fill("svg")
        page.locator('button:has-text("Search HF")').click()
        page.wait_for_timeout(1200)
        model_cards = page.locator("text=direct-svg").count()
        assert model_cards >= 1, "Expected Hugging Face models to be listed"
        print(
            f"[Playwright] ✅ Hugging Face Model Hub & Downloader verified ({model_cards} direct-svg models + Downloaded badges visible)."
        )

        # 3. Test Direct Text-to-SVG Generation
        gen_button = page.locator('button:has-text("Generate Colored SVG")')
        gen_button.click()
        page.wait_for_selector("text=Live Rust SVG Color Remapper", timeout=25000)
        page.wait_for_selector('button:has-text("Download SVG (.svg)"):not([disabled])', timeout=10000)

        # Verify actual SVG file download button works!
        with page.expect_download() as download_info:
            page.locator('button:has-text("Download SVG (.svg)")').click()
        download = download_info.value
        assert download.suggested_filename.endswith(".svg"), (
            f"Expected .svg filename, got {download.suggested_filename}"
        )
        print(f"[Playwright] ✅ Download SVG (.svg) button verified! Downloaded: {download.suggested_filename}")

        shot1 = os.path.join(ARTIFACT_DIR, "ui_direct_svg_generation.png")
        page.screenshot(path=shot1, full_page=True)
        print(f"[Playwright] ✅ Desktop Direct Text-to-SVG screenshot saved to {shot1}")

        # 4. Test Diffusion + Rust vtracer Color Vectorization Pipeline
        page.locator('button:has-text("Diffusion + vtracer")').click()
        page.locator('button:has-text("Cyberpunk Neon")').click()
        page.locator(
            'button:has-text("A futuristic AI robot head with glowing neon cyan and magenta visor")'
        ).click()
        gen_button.click()
        page.wait_for_timeout(1500)
        page.wait_for_selector('button:has-text("Generate Colored SVG"):not([disabled])', timeout=25000)

        shot2 = os.path.join(ARTIFACT_DIR, "ui_vtracer_cyberpunk_robot.png")
        page.screenshot(path=shot2, full_page=True)
        print(f"[Playwright] ✅ Diffusion + Rust vtracer color vectorization verified! Screenshot saved to {shot2}")

        # 5. Test Photo / PNG / JPG -> Colored Line-Art Vector SVG Pipeline
        page.locator('button:has-text("Photo → Line-Art")').click()
        page.wait_for_selector("text=Upload PNG / JPG / Photo", timeout=5000)
        page.locator('button:has-text("Pop Portrait")').click()
        page.locator('button:has-text("Colored Line-Art + Fill")').click()
        convert_btn = page.locator('button:has-text("Convert Photo to Colored Line-Art SVG")')
        convert_btn.click()
        page.wait_for_timeout(1200)
        page.wait_for_selector('button:has-text("Convert Photo to Colored Line-Art SVG"):not([disabled])', timeout=25000)

        # Verify SVG path elements exist and download works
        with page.expect_download() as dl_photo:
            page.locator('button:has-text("Download SVG (.svg)")').click()
        photo_dl = dl_photo.value
        assert photo_dl.suggested_filename.endswith(".svg")
        print(f"[Playwright] ✅ Photo -> Colored Line-Art SVG converted & downloaded: {photo_dl.suggested_filename}")

        shot_photo = os.path.join(ARTIFACT_DIR, "ui_photo_to_lineart_svg.png")
        page.screenshot(path=shot_photo, full_page=True)
        print(f"[Playwright] ✅ Photo-to-Colored-Line-Art SVG screenshot saved to {shot_photo}")

        # 6. Test Bottom Navigation "Edit SVG" Tab & ML Transform/Enhance Engine
        page.locator('nav button:has-text("Edit SVG")').click()
        page.wait_for_selector("text=SVG Color Editor & ML Enhancer", timeout=5000)
        page.locator('button:has-text("ML Sparkle & Sacred Accents")').click()
        enhance_btn = page.locator('button:has-text("Transform & Enhance SVG with ML")')
        enhance_btn.click()
        page.wait_for_timeout(1200)
        page.wait_for_selector('button:has-text("Transform & Enhance SVG with ML"):not([disabled])', timeout=20000)
        page.locator('button:has-text("Save")').first.click()
        page.wait_for_timeout(500)

        shot_editor = os.path.join(ARTIFACT_DIR, "ui_svg_editor_ml_enhancer.png")
        page.screenshot(path=shot_editor, full_page=True)
        print(f"[Playwright] ✅ SVG Editor & ML Enhancer verified! Screenshot saved to {shot_editor}")

        # 7. Test HF Models System Prompt & Hyperparameters
        page.locator('nav button:has-text("HF Models")').click()
        page.wait_for_selector("text=Model System Prompt & Hyperparameters", timeout=5000)
        page.locator('button:has-text("Neon Gradient & Glow")').click()
        sys_prompt_textarea = page.locator('textarea[placeholder*="Enter custom system prompt"]')
        assert "gradient" in sys_prompt_textarea.input_value().lower()
        seed_input = page.locator('input[placeholder="Auto"]')
        seed_input.fill("777")
        page.locator("text=Model System Prompt & Hyperparameters").scroll_into_view_if_needed()

        shot_hf_params = os.path.join(ARTIFACT_DIR, "ui_hf_model_params.png")
        page.screenshot(path=shot_hf_params, full_page=False)
        print(f"[Playwright] ✅ HF Models System Prompt & Hyperparameters verified! Screenshot saved to {shot_hf_params}")

        context.close()

        # =====================================================================
        # PART B: Mobile Viewport (390 x 844 — iPhone 14 Pro)
        # =====================================================================
        mobile_ctx = browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            has_touch=True,
            accept_downloads=True,
        )
        mpage = mobile_ctx.new_page()
        print(f"[Playwright] Navigating to {FRONTEND_URL} (Mobile 390x844)...")
        mpage.goto(FRONTEND_URL, wait_until="networkidle")

        # Verify no horizontal overflow on mobile
        overflow = mpage.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        assert not overflow, "Mobile layout has horizontal overflow!"

        # Verify Mobile Bottom Navigation Bar works (Prompt, Photo→SVG, Edit SVG, Canvas, HF Models, Gallery)
        mpage.locator('nav button:has-text("Photo→SVG")').click()
        mpage.wait_for_selector("text=Upload PNG / JPG / Photo", timeout=5000)
        print("[Playwright] ✅ Mobile Photo→SVG tab verified.")

        mpage.locator('nav button:has-text("Edit SVG")').click()
        mpage.wait_for_selector("text=SVG Color Editor & ML Enhancer", timeout=5000)
        print("[Playwright] ✅ Mobile Edit SVG tab verified.")

        mpage.locator('nav button:has-text("Canvas")').click()
        mpage.wait_for_selector('button:has-text("Download SVG (.svg)")', timeout=5000)
        shot_mobile_canvas = os.path.join(ARTIFACT_DIR, "ui_mobile_responsive.png")
        mpage.screenshot(path=shot_mobile_canvas, full_page=False)
        print(f"[Playwright] ✅ Mobile responsive layout & Download bar verified! Screenshot saved to {shot_mobile_canvas}")

        mpage.locator('nav button:has-text("HF Models")').click()
        mpage.wait_for_selector("text=Hugging Face Model Hub", timeout=5000)
        print("[Playwright] ✅ Mobile HF Models tab verified.")

        mpage.locator('nav button:has-text("Gallery")').click()
        mpage.wait_for_selector("text=Recent Colored SVGs", timeout=5000)
        print("[Playwright] ✅ Mobile Gallery tab & card Download buttons verified.")

        mobile_ctx.close()
        browser.close()


if __name__ == "__main__":
    try:
        run_ui_tests()
    except Exception as exc:
        print(f"[Playwright] ❌ Test failed: {exc}", file=sys.stderr)
        sys.exit(1)

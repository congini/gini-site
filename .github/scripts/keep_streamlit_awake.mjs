import { chromium } from "playwright-core";

const appUrl = process.env.STREAMLIT_APP_URL;

if (!appUrl) {
  throw new Error("STREAMLIT_APP_URL is required.");
}

const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

try {
  const page = await browser.newPage();
  const response = await page.goto(appUrl, {
    waitUntil: "domcontentloaded",
    timeout: 120_000,
  });

  if (!response || !response.ok()) {
    throw new Error(`App returned HTTP ${response?.status() ?? "unknown"}.`);
  }

  const wakeButton = page.getByRole("button", {
    name: /get this app back up/i,
  });

  if (await wakeButton.isVisible({ timeout: 5_000 }).catch(() => false)) {
    console.log("The app was asleep; requesting a wake-up.");
    await wakeButton.click();
    await page.waitForFunction(
      () => !/this app has gone to sleep/i.test(document.body.innerText),
      undefined,
      { timeout: 120_000 },
    );
  }

  // Keep a real browser session open long enough for Streamlit's frontend to
  // connect to the app, which is more representative than a health-check ping.
  await page.waitForTimeout(30_000);

  const bodyText = await page.locator("body").innerText();
  if (/this app has gone to sleep/i.test(bodyText)) {
    throw new Error("The app is still showing Streamlit's sleep page.");
  }

  console.log(`Visited ${page.url()} (${await page.title()})`);
} finally {
  await browser.close();
}

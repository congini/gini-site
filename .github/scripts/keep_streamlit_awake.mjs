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

  // isVisible() returns immediately and ignores its timeout option. Streamlit's
  // sleep page hydrates the wake button asynchronously, so explicitly wait for
  // it instead of racing the page render.
  const wakeButtonAppeared = await wakeButton
    .waitFor({ state: "visible", timeout: 30_000 })
    .then(() => true)
    .catch(() => false);

  if (wakeButtonAppeared) {
    console.log("The app was asleep; requesting a wake-up.");
    await wakeButton.click({ timeout: 30_000 });
    await page.waitForFunction(
      () => !/this app has gone to sleep/i.test(document.body.innerText),
      undefined,
      { timeout: 180_000 },
    );
  } else {
    console.log("No sleep-page wake button appeared; verifying the running app.");
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

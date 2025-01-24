import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


def build_nba_url(season, per_mode="PerGame", dir="A", sort="TEAM_ABBREVIATION"):
    """Build the NBA stats URL dynamically with optional parameters."""
    base_url = "https://www.nba.com/stats/players/traditional"
    params = f"?PerMode={per_mode}&Season={season}&dir={dir}&sort={sort}"
    return base_url + params


def select_all_option(driver):
    """Select the 'All' option in the dropdown to display all stats on one page."""
    try:
        dropdown_xpath = "/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select"
        all_option_xpath = "//option[text()='All']"

        # Wait for the dropdown to be interactable
        dropdown = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, dropdown_xpath))
        )
        print("Dropdown located and clickable.")

        # Ensure the dropdown is visible
        driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)

        # Select "All" using JavaScript and trigger the change event
        driver.execute_script("""
            let dropdown = arguments[0];
            dropdown.value = 'All';
            dropdown.dispatchEvent(new Event('change'));
        """, dropdown)
        print("Selected 'All' option successfully.")

        # Wait for table rows to load
        WebDriverWait(driver, 30).until(
            lambda d: len(d.find_elements(By.XPATH, "//table[contains(@class, 'Crom_table')]/tbody/tr")) > 50
        )
        time.sleep(5)  # Additional buffer for the table to load
        print("Table updated after selecting 'All'.")

    except Exception as e:
        print(f"Error selecting 'All' option: {e}")
        raise


def scrape_nba_stats_for_season(season, output_file):
    """Scrape all players' stats for a given season."""
    driver = webdriver.Chrome()  # Ensure ChromeDriver is in PATH
    try:
        # Build the URL
        url = build_nba_url(season)
        print(f"Accessing URL for season {season}: {url}")
        driver.get(url)

        # Select the "All" option from the dropdown menu
        select_all_option(driver)

        # Locate and process the table
        table_xpath = "//*[@id='__next']/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table"
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, table_xpath))
        )
        print("Table located.")
        table = driver.find_element(By.XPATH, table_xpath)

        # Extract headers
        headers = [header.text.strip() for header in table.find_elements(By.TAG_NAME, "th") if header.text.strip()]
        print(f"Headers: {headers}")

        # Extract rows and align with headers
        rows = []
        for row in table.find_elements(By.CSS_SELECTOR, "tbody tr"):
            cells = [cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td")]
            if len(cells) < len(headers):
                cells.extend([None] * (len(headers) - len(cells)))  # Fill missing cells
            elif len(cells) > len(headers):
                headers.extend([f"EXTRA_COLUMN_{i+1}" for i in range(len(cells) - len(headers))])  # Add extra columns
            rows.append(cells)

        print(f"Number of rows extracted: {len(rows)}")

        if not rows:
            print(f"No data found for season {season}.")
            return

        # Save data to CSV
        df = pd.DataFrame(rows, columns=headers)
        df["Season"] = season  # Add season column
        df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))
        print(f"Data for season {season} saved successfully.")

    except Exception as e:
        print(f"Error during scraping for season {season}: {e}")

    finally:
        driver.quit()


# Generate season strings from 2003-04 to 2024-25
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2003, 2025)]
output_file = "nba_allstats.csv"

# Start scraping
for season in seasons:
    scrape_nba_stats_for_season(season, output_file)

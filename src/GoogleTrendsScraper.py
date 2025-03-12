import pandas as pd
import time
import numpy as np
import math
from playwright.sync_api import sync_playwright, TimeoutError
import os
from datetime import datetime, timedelta
from functools import reduce
import re
import tempfile
import uuid
import traceback
import random


# Name of the download file created by Google Trends
NAME_DOWNLOAD_FILE = 'multiTimeline.csv'
# Max number of consecutive daily observations scraped in one go
MAX_NUMBER_DAILY_OBS = 200
# Max number of simultaneous keywords scraped
MAX_KEYWORDS = 5


def scale_trend(data_daily, data_all, frequency):
    """
    Function that rescales the data at daily frequency using data at lower frequency
    Args:
        data_daily: pandas.DataFrame of daily trend data
        data_all: pandas.DataFrame of the trend data over the entire sample at a lower frequency
        frequency: frequency of 'data_all'

    Returns: pandas.DataFrame

    """
    # Ensure that all data is numeric:
    data_daily = data_daily.replace('<1', 0.5).astype('float')
    data_all = data_all.replace('<1', 0.5).astype('float')
    # Compute factor by distinguishing between the frequency of the data at the lower frequency (weekly or monthly)
    if frequency == "Weekly":
        factor_dates = pd.date_range(next_weekday(data_daily.index[0], 0),
                                     previous_weekday(data_daily.index[-1], 6), freq='D')
        data_daily_weekly = data_daily.loc[factor_dates].resample('W').sum()
        factor = data_all.loc[data_daily_weekly.index] / data_daily_weekly
    elif frequency == "Monthly":
        factor_dates = pd.date_range(ceil_start_month(data_daily.index[0]),
                                     floor_end_month(data_daily.index[-1]), freq='D')
        data_daily_monthly = data_daily.loc[factor_dates].resample(
            'M', label="left", loffset=timedelta(1)).sum()
        factor = data_all.loc[data_daily_monthly.index] / data_daily_monthly
    # Transform the factor from a pandas.DataFrame to a flat numpy.array
    factor = np.array(factor).flatten()
    # Remove all factor entries for which either of the series is zero
    factor = factor[factor != 0]
    factor = factor[np.isfinite(factor)]
    # Rescale and return the daily trends
    return data_daily * np.median(factor)


def concat_data(data_list, data_all, keywords, frequency):
    """
    Function that merge the DataFrames obtained from different scrapes of GoogleTrends. The DataFrames are collected in
    a list (ordered chronologically), with the last and first observation of two consecutive DataFrames being from the
    same day.
    Args:
        data_list: list of pandas DataFrame objects
        data_all: pandas.DataFrame of trend data over the entire period for the same keywords
        keywords: list of the keywords for which GoogleTrends has been scraped
        frequency:

    Returns: pandas DataFrame with a 'Date' column and a column for each keyword in 'keywords'

    """
    # Remove trend subperiods for which no data has been found
    data_list = [data for data in data_list if data.shape[0] != 0]
    # Rescale the daily trends based on the data at the lower frequency
    data_list = [scale_trend(x, data_all, frequency) for x in data_list]
    # Combine the single trends that were scraped:
    data = reduce((lambda x, y: x.combine_first(y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    # Rename the columns
    data.columns = keywords
    return data


def merge_two_keyword_chunks(data_first, data_second):
    """
    Given two data frame objects with same index and one overlapping column (keyword), a scaling factor
    is determined and the data frames are merged, where the second data frame is rescaled to match the
    scale of the first data set.

    Args:
        data_first: pandas.DataFrame obtained from the csv-file created by GoogleTrends
        data_second: pandas.DataFrame obtained from the csv-file created by GoogleTrends

    Returns: pandas.DataFrame of the merge and re-scaled input pandas.DataFrame

    """
    common_keyword = data_first.columns.intersection(data_second.columns)[0]
    scaling_factor = np.nanmedian(
        data_first[common_keyword] / data_second[common_keyword])
    data_second = data_second.apply(lambda x: x * scaling_factor)
    data = pd.merge(data_first, data_second.drop(
        common_keyword, axis=1), left_index=True, right_index=True)
    return data


def merge_keyword_chunks(data_list):
    """
    Merge a list of pandas.DataFrame objects with the same index and one overlapping column by appropriately
    re-scaling.

    Args:
        data_list: list of pandas.DataFrame objects to be merged

    Returns: pandas.DataFrame objects of the merged data sets contained in the input list

    """
    # Iteratively merge the DataFrame objects in the list of data
    data = reduce((lambda x, y: merge_two_keyword_chunks(x, y)), data_list)
    # Find the maximal value across keywords and time
    max_value = data.max().max()
    # Rescale the trends by the maximal value, i.e. such that the largest value across keywords and time is 100
    data = 100 * data / max_value
    return data


def adjust_date_format(date, format_in, format_out):
    """
    Converts a date-string from one format to another
    Args:
        date: datetime as a string
        format_in: format of 'date'
        format_out: format to which 'date' should be converted

    Returns: date as a string in the new format

    """
    return datetime.strptime(date, format_in).strftime(format_out)


def get_chunks(list_object, chunk_size):
    """
    Generator that divides a list into chunks. If the list is divided in two or more chunks, two consecutive chunks
    have one common element.

    Args:
        list_object: iterable
        chunk_size: size of each chunk as an integer

    Returns: iterable list in chunks with one overlapping element

    """
    size = len(list_object)
    if size <= chunk_size:
        yield list_object
    else:
        chunks_nb = math.ceil(size / chunk_size)
        iter_ints = range(0, chunks_nb)
        for i in iter_ints:
            j = i * chunk_size
            if i + 1 < chunks_nb:
                k = j + chunk_size
                yield list_object[max(j - 1, 0):k]
            else:
                yield list_object[max(j - 1, 0):]


def previous_weekday(date, weekday):
    """
    Function that rounds a date down to the previous date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = date.weekday() - weekday
    if delta < 0:
        delta += 7
    return date + timedelta(days=-int(delta))


def next_weekday(date, weekday):
    """
    Function that rounds a date up to the next date of the desired weekday
    Args:
        date: a datetime.date or datetime.datetime object
        weekday: the desired week day as integer (Monday = 0, ..., Sunday = 6)

    Returns: datetime.date or datetime.datetime object

    """
    delta = weekday - date.weekday()
    if delta < 0:
        delta += 7
    return date + timedelta(days=int(delta))


def ceil_start_month(date):
    """
    Ceil date to the start date of the next month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    if date.month == 12:
        date = datetime(date.year + 1, 1, 1)
    else:
        date = datetime(date.year, date.month + 1, 1)
    return date


def floor_end_month(date):
    """
    Floor date to the end of the previous month
    Args:
        date: datetime.datetime object

    Returns: datetime.datetime object

    """
    return datetime(date.year, date.month, 1) + timedelta(days=-1)


class GoogleTrendsScraper:
    def __init__(self, sleep=1, path_driver=None, headless=True, date_format='%Y-%m-%d', debug=True):
        """
        Constructor of the Google-Scraper-Class
        Args:
            sleep: integer number of seconds where the scraping waits (avoids getting blocked and gives the code time
                    to download the data
            path_driver: path as string to where the chrome driver is located (not used with Playwright)
            headless: boolean indicating whether the browser should be displayed or not
            date_format: format in which the date-strings are passed to the object
            debug: boolean indicating whether to save debug screenshots
        """
        # Current directory
        self.dir = os.getcwd()
        # Define download folder for browser:
        self.download_path = tempfile.TemporaryDirectory()
        # Define the path to the downloaded csv-files (this is where the trends are saved)
        self.filename = os.path.join(self.download_path.name, NAME_DOWNLOAD_FILE)
        # Whether the browser should be opened in headless mode
        self.headless = headless
        # Path to the driver of Google Chrome (not used with Playwright)
        self.path_driver = path_driver
        # Initialize the browser variable
        self.browser = None
        self.page = None
        self.playwright = None
        self.context = None
        # Sleep time used during the scraping procedure
        self.sleep = sleep
        # Maximal number of consecutive days scraped
        self.max_days = MAX_NUMBER_DAILY_OBS
        # Format of the date-strings
        self.date_format = date_format
        # Format of dates used by google
        self._google_date_format = '%Y-%m-%d'
        # Debug mode
        self.debug = debug
        # Screenshot counter
        self.screenshot_counter = 0
        # 常见的用户代理字符串列表
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0'
        ]
        # Launch the browser
        self.start_browser()

    def start_browser(self):
        """
        Method that initializes a Playwright browser
        """
        # If the browser is already running, do not start a new one
        if self.browser is not None:
            print('Browser already running')
            return
            
        try:
            # Start Playwright and launch browser with minimal options
            self.playwright = sync_playwright().start()
            
            # 随机选择一个用户代理
            user_agent = random.choice(self.user_agents)
            print(f"Using user agent: {user_agent}")
            
            # 使用Firefox浏览器代替Chromium
            print("Launching Firefox browser...")
            self.browser = self.playwright.firefox.launch(
                headless=self.headless
            )
            
            # 创建一个带有用户代理的浏览器上下文
            self.context = self.browser.new_context(
                accept_downloads=True,
                user_agent=user_agent,
                viewport={'width': 1920, 'height': 1080},
                locale='en-US'
            )
            
            # 创建一个新页面
            self.page = self.context.new_page()
            
            # 配置下载行为
            self.page.on("download", lambda download: download.save_as(
                os.path.join(self.download_path.name, NAME_DOWNLOAD_FILE)))
                
            # 先访问Google首页，设置正确的引用来源
            print("Visiting Google homepage first...")
            self.page.goto('https://www.google.com/', wait_until='networkidle')
            time.sleep(2)
                
            print("Browser started successfully")
            
        except Exception as e:
            print(f"Error starting browser: {str(e)}")
            traceback.print_exc()
            # 尝试清理任何可能已创建的资源
            self.quit_browser()
            raise

    def quit_browser(self):
        """
        Method that closes the existing browser
        """
        try:
            if self.page is not None:
                try:
                    self.page.close()
                except Exception as e:
                    print(f"Error closing page: {str(e)}")
                self.page = None
                
            if self.context is not None:
                try:
                    self.context.close()
                except Exception as e:
                    print(f"Error closing context: {str(e)}")
                self.context = None
                
            if self.browser is not None:
                try:
                    self.browser.close()
                except Exception as e:
                    print(f"Error closing browser: {str(e)}")
                self.browser = None
                
            if self.playwright is not None:
                try:
                    self.playwright.stop()
                except Exception as e:
                    print(f"Error stopping playwright: {str(e)}")
                self.playwright = None
                
        except Exception as e:
            print(f"Error during browser cleanup: {str(e)}")
            traceback.print_exc()

    def take_screenshot(self, name_suffix=""):
        """
        Take a screenshot of the current page for debugging purposes
        """
        if not self.debug or not self.page:
            return
            
        try:
            filename = f"debug_screenshot_export_{self.screenshot_counter}{name_suffix}.png"
            self.page.screenshot(path=filename)
            print(f"Screenshot saved to {filename}")
            self.screenshot_counter += 1
        except Exception as e:
            print(f"Failed to take screenshot: {str(e)}")

    def get_trends(self, keywords, start, end, region=None, category=None):
        """
        Function that starts the scraping procedure and returns the Google Trend data.
        Args:
            keywords: list or string of keyword(s)
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: pandas DataFrame with a 'Date' column and a column containing the trend for each keyword in 'keywords'

        """
        # If only a single keyword is given, i.e. as a string and not as a list, put the single string into a list
        if not isinstance(keywords, list):
            keywords = [keywords]
        # Convert the date strings to Google's format:
        start = adjust_date_format(
            start, self.date_format, self._google_date_format)
        end = adjust_date_format(
            end, self.date_format, self._google_date_format)
        # Create datetime objects from the date-strings:
        start_datetime = datetime.strptime(start, self._google_date_format)
        end_datetime = datetime.strptime(end, self._google_date_format)
        data_keywords_list = []
        for keywords_i in get_chunks(keywords, MAX_KEYWORDS):
            # Get the trends over the entire sample:
            url_all_i = self.create_url(keywords_i,
                                        previous_weekday(start_datetime, 0), next_weekday(
                                            end_datetime, 6),
                                        region, category)
            data_all_i, frequency_i = self.get_data(url_all_i)
            
            # 如果数据为空，返回空的DataFrame
            if data_all_i.empty:
                print("获取到的数据为空，返回空DataFrame")
                return pd.DataFrame()
                
            # If the data for the entire sample is already at the daily frequency we are done. Otherwise we need to
            # get the trends for sub-periods
            if frequency_i == 'Daily':
                data_i = data_all_i
            else:
                # Iterate over the URLs of the sub-periods and retrieve the Google Trend data for each
                data_time_list = []
                for url in self.create_urls_subperiods(keywords_i, start_datetime, end_datetime, region, category):
                    data_time_list.append(self.get_data(url)[0])
                    
                # 检查是否有数据
                if not data_time_list:
                    print("子时间段没有数据，返回空DataFrame")
                    return pd.DataFrame()
                    
                # Concatenate the so obtained set of DataFrames to a single DataFrame
                data_i = concat_data(
                    data_time_list, data_all_i, keywords_i, frequency_i)
                    
            # Add the data for the current list of keywords to a list
            data_keywords_list.append(data_i)
            
        # 检查是否有数据
        if not data_keywords_list:
            print("没有关键词数据，返回空DataFrame")
            return pd.DataFrame()
            
        # Merge the multiple keyword chunks
        try:
            data = merge_keyword_chunks(data_keywords_list)
            
            # Cut data to return only the desired period:
            data = data.loc[data.index.isin(pd.date_range(
                start_datetime, end_datetime, freq='D'))]
                
            return data
        except Exception as e:
            print(f"合并关键词数据时出错: {str(e)}")
            
            # 如果只有一个关键词，直接返回第一个数据
            if len(data_keywords_list) == 1:
                return data_keywords_list[0]
                
            # 否则尝试手动合并
            try:
                print("尝试手动合并数据...")
                result = pd.DataFrame()
                for df in data_keywords_list:
                    if result.empty:
                        result = df
                    else:
                        # 尝试合并列
                        for col in df.columns:
                            if col not in result.columns:
                                result[col] = df[col]
                return result
            except Exception as merge_error:
                print(f"手动合并数据时出错: {str(merge_error)}")
                # 如果所有尝试都失败，返回第一个数据
                if data_keywords_list:
                    return data_keywords_list[0]
                return pd.DataFrame()

    def create_urls_subperiods(self, keywords, start, end, region=None, category=None):
        """
        Generator that creates an iterable of URLs that open the Google Trends for a series of sub periods
        Args:
            keywords: list of string keywords
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            start: start date as a string
            end: end date as a string
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: iterable of URLs for Google Trends for sub periods of the entire period defined by 'start' and 'end'

        """
        # Calculate number of sub-periods and their respective length:
        num_subperiods = np.ceil(((end - start).days + 1) / self.max_days)
        num_days_in_subperiod = np.ceil(
            ((end - start).days + 1) / num_subperiods)
        for x in range(int(num_subperiods)):
            start_sub = start + timedelta(days=x * num_days_in_subperiod)
            end_sub = start + \
                timedelta(days=(x + 1) * num_days_in_subperiod - 1)
            if end_sub > end:
                end_sub = end
            if start_sub < end:
                yield self.create_url(keywords, start_sub, end_sub, region=region, category=category)

    def create_url(self, keywords, start, end, region=None, category=None):
        """
        Creates a URL for Google Trends
        Args:
            keywords: list of string keywords
            start: start date as a string
            end: end date as a string
            region: string indicating the region for which the trends are computed, default is None (Worldwide trends)
            category: integer indicating the category (e.g. 7 is the category "Finance")

        Returns: string of the URL for Google Trends of the given keywords over the time period from 'start' to 'end'

        """
        # Replace the '+' symbol in a keyword with '%2B'
        keywords = [re.sub(r'[+]', '%2B', kw) for kw in keywords]
        # Replace white spaces in a keyword with '%20'
        keywords = [re.sub(r'\s', '%20', kw) for kw in keywords]
        # Define main components of the URL
        base = "https://trends.google.com/trends/explore"
        geo = f"geo={region}&" if region is not None else ""
        query = f"q={','.join(keywords)}"
        cat = f"cat={category}&" if category is not None else ""
        # Format the datetime objects to strings in the format used by google
        start_string = datetime.strftime(start, self._google_date_format)
        end_string = datetime.strftime(end, self._google_date_format)
        # Define the date-range component for the URL
        date = f"date={start_string}%20{end_string}"
        # Construct the URL
        url = f"{base}?{cat}{date}&{geo}{query}"
        return url

    def get_data(self, url):
        """
        Method that retrieves for a specific URL the Google Trend data. Note that this is done by downloading a csv-file
        which is then loaded and stored as a pandas.DataFrame object
        Args:
            url: URL for the trend to be scraped as a string

        Returns: a pandas.DataFrame object containing the trends for the given URL
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查浏览器是否正常运行
                if self.page is None or self.browser is None:
                    print("Browser not running, restarting...")
                    self.quit_browser()
                    self.start_browser()
                    self.take_screenshot("_after_browser_restart")
                
                # Navigate to the URL
                self.go_to_url(url)
                
                # Take screenshot after navigation
                self.take_screenshot("_after_navigation")
                
                # Sleep the code by the defined time plus a random number of seconds between 0s and 2s. This should
                # reduce the likelihood that Google detects us as a scraper
                time.sleep(self.sleep * (1 + np.random.rand()))
                
                # Try to find the button and click it
                try:
                    # Wait for the line chart to be visible
                    print("Waiting for chart to load...")
                    self.page.wait_for_selector("widget[type='fe_line_chart']", timeout=30000)
                    
                    # Take screenshot after chart is loaded
                    self.take_screenshot("_chart_loaded")
                    
                    # 等待导出按钮可见
                    print("Waiting for export button...")
                    self.page.wait_for_selector('.widget-actions-item.export', timeout=10000)
                    
                    # Click the export button
                    print("Clicking export button...")
                    self.page.click('.widget-actions-item.export')
                    
                    # Take screenshot after clicking export
                    self.take_screenshot("_after_export_click")
                    
                    # Wait for download to complete
                    print("Waiting for download to complete...")
                    time.sleep(self.sleep * (1 + np.random.rand()))
                    
                    # Check if file exists
                    wait_time = 0
                    while not os.path.exists(self.filename) and wait_time < 10:
                        time.sleep(1)
                        wait_time += 1
                        print(f"Waiting for file to appear... {wait_time}/10")
                        
                    if not os.path.exists(self.filename):
                        self.take_screenshot("_download_failed")
                        raise Exception("Download failed: File not found")
                        
                except TimeoutError as te:
                    # If we get a timeout, check for 429 error
                    try:
                        self.take_screenshot("_timeout_error")
                        print(f"Timeout error: {str(te)}")
                        
                        content = self.page.content()
                        if "429" in content:
                            self.take_screenshot("_429_error")
                            print("Encountered 429 error (Too Many Requests). Refreshing and retrying...")
                            self.page.reload()
                            time.sleep(self.sleep * 2)  # Wait longer after a 429
                            self.take_screenshot("_after_429_refresh")
                            retry_count += 1
                            continue
                        else:
                            # 检查页面内容，查找其他可能的错误
                            if "not available" in content.lower():
                                self.take_screenshot("_data_not_available")
                                print("Data not available for this query")
                                # 返回空数据框
                                return pd.DataFrame(), 'Daily'
                            else:
                                raise Exception("Timeout waiting for chart to load")
                    except Exception as e:
                        # If page is closed or another error occurs
                        print(f"Error checking page content: {str(e)}")
                        # Restart browser if it's closed
                        if "Target page, context or browser has been closed" in str(e):
                            print("Browser was closed. Restarting...")
                            self.quit_browser()
                            self.start_browser()
                            self.take_screenshot("_after_browser_restart")
                            retry_count += 1
                            continue
                        else:
                            raise
                
                # Load the data from the csv-file as pandas.DataFrame object
                print("Loading data from CSV file...")
                try:
                    # 保存文件内容以便调试
                    if os.path.exists(self.filename):
                        with open(self.filename, 'r') as f:
                            file_content = f.read()
                            print(f"CSV file content preview: {file_content[:200]}...")
                    
                    # 读取CSV文件
                    data = pd.read_csv(self.filename, skiprows=1)
                    print(f"CSV columns: {data.columns.tolist()}")
                    
                    # 确定频率并设置日期索引
                    frequency = 'Unknown'
                    
                    # 检查列名以确定频率
                    date_column = None
                    for col in ['Day', 'Week', 'Month']:
                        if col in data.columns:
                            date_column = col
                            frequency = col + 'ly' if col != 'Month' else 'Monthly'
                            break
                    
                    # 如果找到日期列，设置为索引
                    if date_column:
                        print(f"Found date column: {date_column}, frequency: {frequency}")
                        data[date_column] = pd.to_datetime(data[date_column])
                        data = data.set_index(date_column)
                    else:
                        # 如果没有找到标准日期列，尝试查找其他可能的日期列
                        print(f"Standard date column not found. Available columns: {data.columns.tolist()}")
                        # 尝试查找包含"date"或"time"的列
                        for col in data.columns:
                            if 'date' in col.lower() or 'time' in col.lower():
                                print(f"Using {col} as date column")
                                try:
                                    data[col] = pd.to_datetime(data[col])
                                    data = data.set_index(col)
                                    frequency = 'Daily'  # 默认假设为每日数据
                                    break
                                except:
                                    print(f"Failed to convert {col} to datetime")
                        
                        # 如果仍然没有找到日期列，使用默认索引
                        if date_column is None:
                            print("No date column found, using default index")
                            frequency = 'Daily'  # 默认假设为每日数据
                
                except Exception as csv_error:
                    self.take_screenshot("_csv_read_error")
                    print(f"Error reading CSV file: {str(csv_error)}")
                    # 检查文件是否存在且大小大于0
                    if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
                        # 尝试读取文件内容并打印
                        with open(self.filename, 'r') as f:
                            print(f"CSV file content: {f.read()}")
                    raise
                
                print(f"Data loaded successfully with frequency: {frequency}")
                
                # Sleep again
                time.sleep(self.sleep * (1 + np.random.rand()))
                
                # Delete the file
                while os.path.exists(self.filename):
                    try:
                        os.remove(self.filename)
                    except:
                        pass
                        
                return data, frequency
                
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                traceback.print_exc()
                
                try:
                    # Take screenshot of the error state
                    self.take_screenshot(f"_error_{retry_count}")
                    
                    # Check if it's a 429 error
                    if "429" in str(e):
                        print("Encountered 429 error (Too Many Requests). Refreshing and retrying...")
                        try:
                            if self.page:
                                self.page.reload()
                                self.take_screenshot("_after_429_refresh")
                        except:
                            # If page reload fails, restart browser
                            self.quit_browser()
                            self.start_browser()
                            self.take_screenshot("_after_browser_restart")
                        time.sleep(self.sleep * 2)  # Wait longer after a 429
                    elif "Target page, context or browser has been closed" in str(e):
                        print("Browser was closed. Restarting...")
                        self.quit_browser()
                        self.start_browser()
                        self.take_screenshot("_after_browser_restart")
                    else:
                        print(f"Retrying... ({retry_count + 1}/{max_retries})")
                        time.sleep(self.sleep)
                except Exception as inner_e:
                    print(f"Error during error handling: {str(inner_e)}")
                    traceback.print_exc()
                    # Restart browser as a last resort
                    try:
                        self.quit_browser()
                        self.start_browser()
                        self.take_screenshot("_after_emergency_restart")
                    except:
                        pass
                
                retry_count += 1
                
        # If we've exhausted all retries
        raise Exception(f"Failed to retrieve data after {max_retries} attempts")

    def go_to_url(self, url):
        """
        Method that navigates in the browser to the given URL
        Args:
            url: URL to which we want to navigate as a string
        """
        if self.page is not None:
            try:
                print(f"Navigating to: {url}")
                
                # 设置额外的HTTP头，包括引用页
                self.page.set_extra_http_headers({
                    'Referer': 'https://trends.google.com/trends/explore',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'same-origin',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                # 导航到URL
                response = self.page.goto(url, wait_until="networkidle", timeout=60000)
                
                # Check for 429 status and handle it
                if response and response.status == 429:
                    self.take_screenshot("_429_status_code")
                    print("Received 429 status code. Refreshing the page...")
                    time.sleep(self.sleep * 2)
                    self.page.reload()
                    self.take_screenshot("_after_429_refresh")
            except Exception as e:
                if "Target page, context or browser has been closed" in str(e):
                    print("Browser was closed. Restarting...")
                    self.quit_browser()
                    self.start_browser()
                    self.take_screenshot("_after_browser_restart")
                    # Try again with the new browser
                    try:
                        self.page.goto(url, wait_until="networkidle", timeout=60000)
                        self.take_screenshot("_after_retry_navigation")
                    except Exception as retry_e:
                        print(f"Error retrying navigation: {str(retry_e)}")
                        self.take_screenshot("_retry_navigation_error")
                elif "429" in str(e):
                    self.take_screenshot("_429_during_navigation")
                    print("Detected 429 error during navigation. Refreshing the page...")
                    try:
                        self.page.reload()
                        self.take_screenshot("_after_429_refresh")
                    except:
                        # If reload fails, restart browser
                        self.quit_browser()
                        self.start_browser()
                        self.take_screenshot("_after_browser_restart")
                else:
                    print(f"Navigation error: {str(e)}")
                    self.take_screenshot("_navigation_error")
        else:
            print('Browser is not running')
            self.start_browser()
            self.take_screenshot("_after_browser_start")
            if self.page:
                try:
                    self.page.goto(url, wait_until="networkidle", timeout=60000)
                    self.take_screenshot("_after_navigation")
                except Exception as e:
                    print(f"Error navigating after browser restart: {str(e)}")
                    self.take_screenshot("_navigation_error_after_restart")

    def __del__(self):
        """
        When deleting an instance of this class, delete the temporary file folder and close the browser
        """
        self.download_path.cleanup()
        self.quit_browser()

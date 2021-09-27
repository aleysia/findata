import numpy as np
import pandas as pd
import quandl
import urllib.request
import requests
import json
import os
from pathlib import Path
import threading
import shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import random
import re
import datetime as dt
from pandas.tseries.offsets import BDay
import pandas_market_calendars as mcal
import yaml
from functools import reduce

import findata.utils as fdutils


class DataImporter:


    def __init__(self, start_date='04/30/2011', end_date=None, save_dir='data/raw', config_file='config.yml',
                 yf_error_msgs=('404 Not Found', 'upstream connect error'),
                 yf_headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                             'content-type': "application/json",
                             'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'}
                ):
        self.start_date = dt.datetime.strptime(start_date, '%m/%d/%Y')
        self.end_date = dt.datetime.now() if end_date is None else dt.datetime.strptime(end_date, '%m/%d/%Y')
        if self.start_date.month==self.end_date.month and \
                self.start_date.day==self.end_date.day and self.start_date.year==self.end_date.year:
            raise ValueError('Start and end date are the same!')
        self.save_dir = os.path.join(save_dir, '')
        if not os.path.exists(self.save_dir):
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.logger = fdutils.new_logger('DataImporter')
        self.yfheaders = yf_headers
        self._parse_config(config_file)
        self.yf_error_msgs = yf_error_msgs


    def get_ticker_list(self, name='sp500', save_dir='data/symbols', overwrite=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, name + '.csv')
        self.ticker_file = filename

        if os.path.exists(filename) and not overwrite:
            self.tickers = pd.read_csv(filename)
        else:
            if name=='sp500':
                # Scrape list of S&P500 companies from Wikipedia
                html = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                soup = BeautifulSoup(html.text, 'lxml')
                table = soup.find('table', {'class': 'wikitable sortable'})

                tickers = []
                for row in table.findAll('tr')[1:]:
                    ticker = row.findAll('td')[0].text
                    ticker = ticker.strip()
                    tickers.append(ticker)

                self.tickers = pd.DataFrame(data={'symbol':tickers, 'prices_exist': True})
            self.tickers.to_csv(filename, index=False)


    def save_ticker_list(self):
        self.tickers.to_csv(self.ticker_file, index=False)


    def _parse_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)


    def firefox_session(self, url, headless=True):
        options = Options()
        options.headless = headless

        fp = webdriver.FirefoxProfile()
        fp.set_preference("browser.download.folderList", 2)
        fp.set_preference("browser.download.manager.showWhenStarting", False)
        fp.set_preference("browser.download.dir", 
                          f'{os.getcwd()}/{self.save_dir}'.replace(r'/','\\'))
        fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")

        self.logger.info(f'Opening {url} in Firefox')

        driver = webdriver.Firefox(options=options, firefox_profile=fp, 
                                   executable_path=self.config['geckodriver']['path'])
        driver.implicitly_wait(10)
        driver.get(url)
        time.sleep(5)

        return driver


    def import_all(self):
        functions = [self.import_dix_gex, self.import_yf_indicators]
        if 'quandl' in self.config.keys():
            functions.append(self.import_quandl)
        if 'investing' in self.config.keys():
            functions.append(self.import_move)
            functions.append(self.import_pmi)

        threads = []
        for f in functions:
            process = threading.Thread(target=f)
            process.start()
            threads.append(process)
        for process in threads:
            process.join()
        self.import_tickers()


    def download_file(self, save_file, url):
        self.logger.info(f'Beginning download of {url} to {self.save_dir}{save_file}')
        try:
            urllib.request.urlretrieve(url, self.save_dir + save_file)
            self.logger.info(f'Finished download of {url}')
        except Exception as e:
            self.logger.debug(f'Exception encountered while downloading {url}:\n' + str(e))


    def import_dix_gex(self, save_file='DIX.csv',
                       url='https://squeezemetrics.com/monitor/static/DIX.csv'):
        self.download_file(save_file, url)


    def login_investing(self, driver):
        # Close stupid popup
        try:
            element = driver.find_element_by_class_name("allow-notifications-popup-close-button")
            ActionChains(driver).move_to_element(element).click().perform()
        except Exception:
            pass
        time.sleep(random.randint(2,4))

        # Other stupid popup
        ActionChains(driver).send_keys('Keys.ESCAPE').perform()

        self.logger.info("Logging into 'www.investing.com'")
        # Login
        element = driver.find_element_by_class_name("login")
        ActionChains(driver).move_to_element(element).click().perform()
        time.sleep(random.randint(2,4))

        # Enter user name
        element = driver.find_element_by_id('loginFormUser_email')
        element.clear()
        # element.send_keys("v.y.chen@gmail.com")
        ActionChains(driver).move_to_element(element).click().perform()
        ActionChains(driver).send_keys(self.config['investing']['email']).perform()
        time.sleep(random.randint(1,3))

        # Enter password
        element = driver.find_element_by_id('loginForm_password')
        element.clear()
        # element.send_keys("BCe4a4mDuwmvp6C")
        ActionChains(driver).move_to_element(element).click().perform()
        ActionChains(driver).send_keys(self.config['investing']['pwd']).perform()
        time.sleep(random.randint(1,3))

        # Click submit
        ActionChains(driver).send_keys(Keys.ENTER).perform()
        time.sleep(random.randint(4,7))

        return driver


    def import_move(self, 
                    default_file='ICE BofAML MOVE Historical Data.csv', save_file='indicator_move.csv',
                    url='https://www.investing.com/indices/ice-bofaml-move-historical-data'):
        driver = self.firefox_session(url=url, headless=False)
        driver = self.login_investing(driver)
        try:
            self.logger.info('Picking start date and downloading MOVE index')
            # Pick date range
            element = driver.find_element_by_id("widgetFieldDateRange")
            ActionChains(driver).move_to_element(element).click().perform()
            time.sleep(random.randint(2,4))

            for i in range(10):
                ActionChains(driver).send_keys(Keys.BACKSPACE).perform()
                time.sleep(0.1)
            ActionChains(driver).send_keys(self.start_date.strftime('%m/%d/%Y')).perform()
            time.sleep(random.randint(1,3))

            # Click apply
            ActionChains(driver).send_keys(Keys.ENTER).perform()
            time.sleep(random.randint(1,3))

            # Save file to Downloads
            element = driver.find_element_by_xpath("//a[@title='Download Data']")
            ActionChains(driver).move_to_element(element).click().perform()
            time.sleep(random.randint(5,10))
                
            self.logger.info('Moving downloaded file from temp directory')
            # Move file to desired directory
            shutil.move(self.save_dir + default_file, self.save_dir + save_file)
        except Exception as e:
            self.logger.debug("Exception occurred while downloading MOVE index':\n" + str(e))
        finally:
            try:
                driver.quit()
            except:
                return None


    def import_pmi(self, save_file='indicator_pmi.csv',
                   url='https://www.investing.com/economic-calendar/manufacturing-pmi-829'):
        driver = self.firefox_session(url=url, headless=False)
        driver = self.login_investing(driver)
        try:
            self.logger.info('Expanding PMI table')
            # Click "Show More" until there is no more to be shown
            while True:
                try:
                    element = driver.find_element_by_xpath("//div[@id='showMoreHistory829']/a")
                    if not element.is_displayed():
                        self.logger.info('Finished expanding PMI table')
                        break
                    element.send_keys(Keys.ENTER)
                    time.sleep(random.uniform(1,5))
                except Exception as e:
                    self.logger.debug('Exception occurred while expanding PMI table:\n' + str(e))
                    break

            self.logger.info('Scraping PMI table')

            page = BeautifulSoup(driver.page_source, 'lxml')

            tab = []
            for tr in page.find_all('table')[0].find_all('tr')[2:]:
                tds = tr.find_all('td')
                row = [tr.text for tr in tds]
                tab.append(row)

            tab = pd.DataFrame(tab, columns=['Date', 'Time', 'Actual', 'Forecast', 'Previous', 'None'])
            tab.drop(columns=['Time','Previous','None'], inplace=True)
            tab.Date = [re.sub(' [(].*','',x) for x in tab.Date]
            tab.replace('\xa0', np.nan,inplace=True)

            tab.to_csv(self.save_dir + save_file, index=False)
        except Exception as e:
            self.logger.debug("Exception occurred while downloading PMI index':\n" + str(e))
        finally:
            try:
                driver.quit()
            except:
                return None


    def download_yf_prices(self, ticker, save_file):
        params = {
            "period1": int(time.mktime(self.start_date.timetuple()))+14400,
            "period2": int(time.mktime(self.end_date.replace(hour=23, minute=59, second=59).timetuple()))+14400,
            "interval": '1d',
            "events": "history",
            "includeAdjustedClose": 'true'
        }
        url = rf"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?" \
            + '&'.join('='.join([k, str(params[k])]) for k in params.keys())
        r = requests.get(url, headers=self.yfheaders)

        if not r.text.startswith(self.yf_error_msgs):
            try:
                with open(save_file, 'w+') as f:
                    f.write(r.text)
            except Exception as e:
                self.logger.debug(f'File save for {ticker} failed in {os.getcwd()}:\n' + str(e))
        else:
            self.logger.debug(f'{ticker} not found in Yahoo Finance')


    def import_tickers(self, skip_existing=False, save_subdir='tickers', suffix=''):
        tickers = self.tickers['symbol']
        tickers = [ticker + suffix for ticker in tickers]

        save_subdir = os.path.join(save_subdir, '')
        if not os.path.exists(self.save_dir + save_subdir):
            os.makedirs(self.save_dir + save_subdir)

        self.logger.info('Downloading tickers from Yahoo Finance')
        for ticker in tickers:
            save_file = f'{self.save_dir}{save_subdir}{ticker}.csv'
            if skip_existing:
                if os.path.exists(save_file):
                    continue
            self.download_yf_prices(ticker, save_file)
            time.sleep(random.uniform(0.5,2))
        self.logger.info('Finished download of tickers from Yahoo Finance')


    def rec2row(self, rrr):
        return pd.DataFrame(data=rrr, index=[0])


    def processRecords(self, rr, data):
        for i in range(0, len(rr)):
            x = self.rec2row(rr[i])
            if not data is None:
                data = data.append(x)
            else:
                data = x
        return data


    def download_yf_chain(self, driver, date, tag, ticker, save_dir):
        try:
            opt_url = 'https://finance.yahoo.com/quote/' + ticker + '/options?p=' + ticker
            driver.get(opt_url)
            page = BeautifulSoup(driver.page_source, 'lxml')
            # Retrieve all options listed, and their associated ID number
            optslist = page.select_one('select.Bd')

            opt_id = list()
            opt_dt = list()
            for o in optslist.select('option'):
                opt_id.append(o['value'])
                opt_dt.append(o.string)
            opts_list = pd.DataFrame(data={'id':opt_id, 'date':opt_dt})

            data = None
            for expdt in opts_list['id']:
                url = 'https://query1.finance.yahoo.com/v7/finance/options/' + ticker + \
                    '?formatted=false&lang=en-US&region=US&date=' + expdt
                r = requests.get(url, headers=self.yfheaders).json()
                time.sleep(random.uniform(0.5,2))

                data = self.processRecords(r['optionChain']['result'][0]['options'][0]['calls'], data)
                data = self.processRecords(r['optionChain']['result'][0]['options'][0]['puts'], data)

            data = data[['strike','openInterest','inTheMoney','impliedVolatility','ask','bid','expiration','contractSymbol']]
            data['strike'] = pd.to_numeric(data['strike'].astype(str).str.replace(',',''), errors='coerce')
            data['openInterest'] = pd.to_numeric(data['openInterest'].astype(str).str.replace(',',''), errors='coerce')
            data['ask'] = pd.to_numeric(data['ask'].astype(str).str.replace(',',''), errors='coerce')
            data['bid'] = pd.to_numeric(data['bid'].astype(str).str.replace(',',''), errors='coerce')
            data['impliedVolatility'] = pd.to_numeric(data['impliedVolatility'].astype(str).str.replace('%','').str.replace(',',''), errors='coerce')/100
            data['type'] = [re.search('[A-Z]+[0-9]+(C|P).*',x).group(1) for x in data['contractSymbol']] 
            data['chain_date'] = date
            data = data.sort_values(by=['strike'])

            save_file = os.path.join(save_dir, date+'.csv') if tag=='daily' \
                else os.path.join(save_dir, date+'_'+tag+'.csv')
            if not os.path.exists(save_dir):
                Path(os.path.join(save_dir)).mkdir(parents=True, exist_ok=True)
            data.to_csv(save_file, index=False)
            return True
        except Exception as e:
            self.logger.debug(f'Error encountered downloading {ticker} options:\n' + str(e))
            return False


    def import_options(self, skip_existing=True, save_subdir='options', chain_date=None, tag='daily'):
        save_subdir = os.path.join(self.save_dir, save_subdir, '')
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)

        if not chain_date:
            chain_date = dt.datetime.today()
        if isinstance(chain_date, str):
            chain_date = dt.datetime.strptime(chain_date, '%Y-%m-%d')
        
        chain_date_str = chain_date.strftime('%Y-%m-%d')
        nyse = mcal.get_calendar('NYSE')
        sch = nyse.schedule(chain_date_str, chain_date_str)
        if tag!='daily' and len(sch)==0:
            self.logger.info('Today is not a business day, skipping intraday options download')
            return None
        if len(sch)==0:
            search_range_start = (chain_date + BDay(-5)).strftime('%Y-%m-%d')
            sch = nyse.schedule(search_range_start, chain_date_str)
            chain_date = sch.index[-1]
        chain_date = chain_date.strftime('%Y-%m-%d')
            

        driver = self.firefox_session('http://finance.yahoo.com')
        for ticker in self.tickers['symbol']:
            filename = os.path.join(save_subdir, ticker, chain_date+'.csv') if tag=='daily' \
                else os.path.join(save_subdir, ticker, chain_date+'_'+tag+'.csv')
            if os.path.isfile(filename):
                self.logger.info(f'Skipping options chain for {ticker}, already downloaded')
                continue

            self.logger.info(f'Beginning options chain download for {ticker}')
            successful = self.download_yf_chain(driver, chain_date, tag, ticker,
                                                os.path.join(save_subdir, ticker))
            if successful:
                self.logger.info(f'Finished downloading options chain for {ticker}')
        
        time.sleep(20)
        driver.quit()


    def import_quandl(self, save_file='Quandl.csv', 
                      indicators={'AAII': 'AAII/AAII_SENTIMENT', 'BTC': 'BCHAIN/MKPRU'},
                      ignore=['AAII_S&P 500 Weekly High', 'AAII_S&P 500 Weekly Low', 'AAII_S&P 500 Weekly Close',
                              'AAII_Bullish 8-week Mov avg', 'AAII_Bullish Average',
                              'AAII_Bullish Average + St. Dev', 'AAII_Bullish Average - St. Dev']):
        results = []
        try:
            for ind in indicators.keys():
                self.logger.info(f'Importing {ind} from Quandl')
                result = quandl.get(indicators[ind], api_key=self.config['quandl']['api-key'])
                result.columns = [f'{ind}_{x}' for x in result.columns if not x in ignore]
                results.append(result)
        except Exception as e:
            self.logger.debug(f'Exception occurred while importing {ind} from Quandl:\n' + str(e))

        try:
            data = reduce(lambda x, y: pd.merge(x, y, how='outer', left_index=True, right_index=True), results)
        except Exception as e:
            self.logger.debug('Exception occurred while concatenanting data from Quandl:\n' + str(e) + f'{results}')
        data.to_csv(self.save_dir + save_file)


    def import_yf_indicators(self, save_subdir='sectors', indicators='all'):
        mapping = {'VIX': 'VXX', 'VVIX': r'%5EVVIX', 'Nasdaq': 'NDX1.DE', 'DXY': 'DX-Y.NYB',
                   'Energy': 'XLE', 'Financials': 'XLF', 'Tech': 'XLK', 'Utilities': 'XLU',
                   'Health': 'XLV', 'Staples': 'XLP', 'Cons. disc.': 'XLU', 'Industrials': 'XLI', 
                   'Communication': 'XLC', 'Materials': 'XLB', 'Real estate': 'XLRE', 'ESG': 'EFIV',
                   'Brent': r'BZ%3DF', 'Gold': r'GC%3DF', 'Dow': r'%5EDJI', 'Russell': r'%5ERUT',
                   'Silver': r'SI%3DF', '10Y yield': r'%5ETNX', 'HY bonds': 'HYG', 'SKEW': r'%5ESKEW',
                   'BTC-USD': 'BTC-USD', 'BTC-CAD': 'BTC-CAD', 'CAD-USD': r'CADUSD%3DX',
                   'Aerospace': 'XAR', 'Banks': 'KBE', 'Biotech': 'XBI', 'Health equip': 'XHE',
                   'Health services': 'XHS', 'Homebuilders': 'XHB', 'Insurance': 'KIE',
                   'Mining': 'XME', 'Natural res': 'GNR', 'Oil exploration': 'XOP',
                   'Pharma': 'XPH', 'Retail': 'XRT', 'Semiconductors': 'XSD', 'Sfotware': 'XSW',
                   'Telecom': 'XTL', 'Transportation': 'XTN', 'Capital marketse': 'KCE',
                   'Infrastructure': 'GII', 'Internet': 'XWEB', 'Momentum': 'MMTM',
                   'Value': 'VLU', 'Emerging markets': 'EEM'}

        if indicators == 'all':
            indicators = mapping.keys()

        save_subdir = os.path.join(save_subdir, '')
        if not os.path.exists(self.save_dir + save_subdir):
            os.makedirs(self.save_dir + save_subdir)

        self.logger.info('Downloading various tickers and ETFs from Yahoo Finance')
        for ind in indicators:
            ticker = mapping[ind]
            save_file = f'{self.save_dir}{save_subdir}{ind}.csv'
            self.download_yf_prices(ticker, save_file)
            time.sleep(random.uniform(0.5,2))
        self.logger.info('Finished download of various tickers and ETFs from Yahoo Finance')

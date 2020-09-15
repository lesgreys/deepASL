# Capstone 2 Proposal:

Ideas are listed in order of data accesibility within the timeframe of the project:
1: Easily Accessible
2: Somewhat Accessible
3: Unlikely Accessible

-----------------

### 1 . Data Easily Accessible 

-----------------
#### Predicting an up or down market day on S&P500. <br>

**Prediction type:** Categorical<br>
**Data type:** Timeseries<br>
**Source:** Yahoo Finance API & Open Source Finance Sites<br>
**Observations/features:** Data would included stock price & volume for S&P 500, DOW30, NASDAQ. Ideally, I would like to include additional data, such as, option interest on indexes and individual stocks that are overweighted/represented within the indexes. Data would go back at a minimum of 10years. Given that the data is timeseries I might test different measures of time; minute, hour, or daily stock price movement. <br>

**Summary:** Using differente measures of the timeseries data for the US Markets Major indices, using logistic regression or NN ML, predict if the following open market day will result in up or down day. I would also implement technical analysis into predictions given that major institutional investors use algorithimic trading and use TA as trigger points for executing trades. Overall, considering stock market predictions tend to follow the equivalence of flipping a coin, a model that can predict with 60% accuracy would be considered a success given that I would be able to execute an investment strategy that would result in an incremental net gain.

-----------------
#### Predict if a college student will graduate or not.
**Prediction type:** Categorical<br>
**Data type:** Stationary<br>
**Source:** data.world, census.gov, NCES, & any open source data<br>
**Observations/features:** Data would included student demographic information i.e. gender, age, race, city, & more. <br>
*description of data by NCES*<br>
The National Center for Education Statistics (NCES) is the primary federal entity for collecting and analyzing data related to education in the U.S. and other nations. NCES is located within the U.S. Department of Education and the Institute of Education Sciences. NCES fulfills a Congressional mandate to collect, collate, analyze, and report complete statistics on the condition of American education; conduct and publish reports; and review and report on education activities internationally.

***Table 326.10.*** Graduation rate from first institution attended for first-time, full-time bachelor's degree-seeking students at 4-year postsecondary institutions, by race/ethnicity, time to completion, sex, control of institution, and acceptance rate: Selected cohort entry years, 1996 through 2008<br>
***Table 326.20.*** Graduation rate from first institution attended within 150 percent of normal time for first-time, full-time degree/certificate-seeking students at 2-year postsecondary institutions, by race/ethnicity, sex, and control of institution: Selected cohort entry years, 2000 through 2011<br>
***Table 326.30.*** Retention of first-time degree-seeking undergraduates at degree-granting postsecondary institutions, by attendance status, level and control of institution, and percentage of applications accepted: Selected years, 2006 to 2014<br>
***Table 326.40.*** Percentage distribution of first-time postsecondary students starting at -2 and 4-year institutions during the 2003-04 academic year, by highest degree attained, enrollment status, and selected characteristics: Spring 2009

**Summary:** Using the features provide within the data described above, using Logistic, RF, GradientBoost ML, predict whether a given student would graduate college or not. This would be a fun personal project considering my personal struggles with the education system. Ideally, I would love to use this to engage students that are at risk of not graduating and assist where necessary to ensure successful educational experience. 

-----------------

### 2. Somewhat Accessible 

-----------------
#### Predict the letter of an image in American Sign Language (ASL)
**Prediction type:** Categorical<br>
**Data type:** Stationary/Images<br>
**Source:** Open source sites with ASL images already populated (GitHub), myself, and google images. Already have a large dataset from Microsoft from an ASL project they started in 2019. <br>
**Observations/features:** Data would include images of all 26 letters in alphabet from A-Z (some letters like j & z require movement). Ideally have 10 images of each letter with different backgrounds to train model.

**Summary:** Using the features provide within the data described above, using Neural Nets/Image Processing predict what letter of the alphabet is displayed in the image. Being raised by deaf parents I have an intimate relationship with the need for ASL to be more interpretable for the masses. This project will be broken down into distinct phases intended to tackle real-world issues for the deaf community. This phase will focus on developing a basic model that accurately predicts what letter of the alphabet a person is signing.

-----------------

### 3. Unlikely Accessible 

-----------------
#### Predict if company IPO will go up or down on NASDAQ.
**Prediction type:** Categorical<br>
**Data type:** Stationary (not timeseries because a company has not started trading)<br>
**Source:** Public domains; USPTO, SEC.gov, NASDAQ IPO filings, open-source news, twitter. <br>
**Observations/features:** Data would long-text/unstructured format of company patent filings, financials from preIPO filings, news articles from public news outlets including Twitter.

**Summary:** Using the features provide within the data described above, use NLP and RNN to produce sentiment analysis and establish a model that can predict whether or not a given IPO will move up or down on the day it debuts on NASDAQ exchange. This project has several challenges since the features needed require extensive data scraping from multiple sources and entails developing 2 predictive models, 1) sentiment analysis predictor 2) IPO debut up or down. I imagine sentiment analysis itself can drastically impact the up/down predictive model. 

-----------------

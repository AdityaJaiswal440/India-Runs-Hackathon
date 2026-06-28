"""Company founding-year reference table.

Used by the honeypot engine to detect candidates whose stated employment
start date pre-dates a company's founding — a reliable indicator of
synthetic data generation artefacts.

Keys are normalised (lowercase, punctuation stripped, legal suffixes removed)
to match the output of the name-cleaning function in the honeypot module.
Extend this dict each quarter as new companies become relevant to the pool.
"""

from __future__ import annotations

COMPANY_FOUNDING_YEARS: dict[str, int] = {
    "google": 1998,
    "netflix": 1997,
    "amazon": 1994,
    "salesforce": 1999,
    "uber": 2009,
    "meta": 2004,
    "adobe": 1982,
    "microsoft": 1975,
    "apple": 1976,
    "linkedin": 2002,
    "infosys": 1981,
    "wipro": 1945,
    "tcs": 1968,
    "swiggy": 2014,
    "razorpay": 2014,
    "cred": 2018,
    "capgemini": 1967,
    "hcl": 1976,
    "zomato": 2008,
    "flipkart": 2007,
    "mindtree": 1999,
    "accenture": 1989,
    "cognizant": 1994,
    "mahindra": 1986,
    "mphasis": 1998,
    "meesho": 2015,
    "nykaa": 2012,
    "inmobi": 2007,
    "byjus": 2011,
    "policybazaar": 2008,
    "ola": 2010,
    "zoho": 1996,
    "vedantu": 2011,
    "paytm": 2010,
    "unacademy": 2015,
    "pharmeasy": 2015,
    "upgrad": 2015,
    "freshworks": 2010,
    "phonepe": 2015,
    "dream11": 2008,
    "glance": 2019,
    "rephraseai": 2019,
    "saarthiai": 2017,
    "sarvamai": 2023,
    "madstreetden": 2013,
    "observeai": 2017,
    "krutrim": 2023,
    "wysa": 2015,
    "haptik": 2013,
    "verloopio": 2015,
    "yellowai": 2016,
    "locobuzz": 2015,
    "niramai": 2016,
    "aganitha": 2017,
    "genpactai": 1997,
}

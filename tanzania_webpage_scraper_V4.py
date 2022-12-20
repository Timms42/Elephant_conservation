"""Author: Liam Timms 44368768
Created: 06/08/2021
Modified: 12/09/2022
This program is for scraping data from Zambia real estate websites
v1: scrapes Tanzania real estate website
v2: - scrapes multiple pages.
    - created function for scraping individual page
v3: takes scraped data as .csv and duplicates it until the cumulative area is the size of Serenget National Park
v4: create dataframe of duplicated real estate data by taking uniform random draws from the real real estate data set
"""

import re
import requests as rq

from bs4 import BeautifulSoup, Tag
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


def remove_featured(ad_list):
    """
    :param: ad_list: (list of str) list of scraped website ads 
    from soup.find_all('div', class_=['product-item__featured', 'product-item__price'])

    :return: ad_list: (list of str) ad_list with featured ads removed
    """

    # List of ads to remove
    featured_ads = []

    for ii in range(len(ad_list)):
        # For each ad, if it is featured, then add it and the next one to the removal list,
        # since featued ads are preceded with text saying that they are featured
        if 'featured' in ad_list[ii]:
            featured_ads.append(ad_list[ii])
            featured_ads.append(ad_list[ii + 1])
        else:
            continue

    # Remove all featured ads from the price_list
    for ad in featured_ads:
        ad_list.remove(ad)

    return ad_list


def scrape_page(url, diction, page_num):
    """
    :param url: (str) website url to scrape
    :param diction: (dict) dictionary for storing real estate prices
    :param page_num: (int) the page number of the website (starts at 0)
    
    :return: diction: (dict) previous dictionary with new prices appended
    """

    if page_num != 0:
        # If this is not the 0th page, then add '?page=X' to the url
        url += '?page={}'.format(page_num)

    # Request html data from website
    site_data = rq.get(url)

    # Parse contents with BeautifulSoup
    soup = BeautifulSoup(site_data.content, 'html.parser')

    # Get the price of each advertisement from the soup (listed in 'product-item__price' class)
    # Also find if ad is featured or not. A featured add will have before it '<div class="product-item__featured">'
    price_list = soup.find_all('div', class_=['product-item__featured', 'product-item__price'])

    # Convert ads in price_list to strings instead of tags
    price_list = [str(ad) for ad in price_list]

    # Remove featured ads from the price_list
    price_list = remove_featured(price_list)

    # ii is the page number, jj is the ad index on the page
    for ii, jj, p in zip([page_num] * len(price_list), range(len(price_list)), price_list):
        # Convert soup Tag into string
        price_string = str(p)

        # Find out where the price is in the string
        digits = [char.isdigit() for char in price_string]  # Check which characters are digits
        if True in digits:
            ind = [jj for jj, x in enumerate(digits) if x == True]  # Indices of digits

            # Extract the price
            price = price_string[ind[0]:ind[-1]]

            # Remove any spaces in the price
            price = price.replace(' ', '')

            # Extract the currency - 3 characters + 1 space before 1st digit
            currency = price_string[ind[0] - 4:ind[0] - 1]

            diction[(ii, jj)] = [price, currency]

        # If there is no price information, ditch this ad and move on
        else:
            continue

    return diction


def convert_area(the_df, hect_2_km2, acre_2_km2):
    """
    Compute size of the areas in km2
    
    :param the_df: (dataframe) dataframe from .csv file 'tanzania_data_new.csv'
    with columns '(Page, ad)', 'Price listed', 'Currency', 'Area listed', 'Area units', 'Per unit area'
    :param hect_2_m2: conversion hectares into km2
    :param acre_2_m2: conversion acres into km2

    :return: df (pandas dataframe) added columns for 'Area'
    """

    # Get df column of the area values
    area_array = np.array(the_df['Area listed'])

    # Which entries are in acres
    area_acre = np.array(the_df['Area units'] == 'acre')
    area_hectare = np.array(the_df['Area units'] == 'hect')  # hectares
    area_m2 = np.array(the_df['Area units'] == 'm2')  # in km2

    m2_2_km2 = 1 / 1000 / 1000  # 1 m2 to km2

    # Construct array of conversion factors into km2 by using some arithmetic with booleans
    conversion_array = area_acre * acre_2_km2 + area_hectare * hect_2_km2 + area_m2 * m2_2_km2

    # Make new column in df that is the converted area values, now in mk2
    the_df['Area'] = area_array * conversion_array
    return the_df


def convert_price(the_df, tsz_2_usd, usd_2022_2_2012):
    """
    Compute the total prices in USD 2012, and the per unit area price in USD 2012
    
    :param the_df: (dataframe) dataframe from .csv file 'tanzania_data_new.csv'
    with columns '(Page, ad)', 'Price listed', 'Currency', 'Area listed', 'Area units', 'Per unit area'
    :param tsz_2_usd: conversion Tanzanian shilling into USD 2022
    :param usd_2022_2_2012: (float) convert USD 2022 to USD 2012

    :return: df (pandas dataframe) added columns for 'Price', 'Price per unit area'
    """

    # Get df column of the area values
    price_array = np.array(the_df['Price listed'])

    # Get df column of the total area values in km2
    area_array = np.array(the_df['Area listed'])
    
    # Compute total price of the area = TZS/unit area * total area * bool(was in price/unit area or not)
    total_price_TZS = price_array * np.array([area_array[ii] if jj == 'yes' else 1
                                              for ii, jj in enumerate(the_df['Per unit area'])])

    # Make new column in df that is price of plot in USD 2012
    the_df['Price'] = total_price_TZS * tsz_2_usd * usd_2022_2_2012

    the_df['Area per unit cost'] = np.array(the_df['Area']) / np.array(the_df['Price'])
    return the_df


def generate_data(the_df, target_area):
    """
    Take dataframe and uniformly draw from it. Add these draws to a new dictionary.
    Index with draw number, e.g. 0, 1, 2, ...
    Add up area until we hit the target area, e.g. Area_cumsum[ii]=Area[ii]+Area_cumsum[ii-1]
    Same with price

    :param the_df: pandas dataframe of real real estate data
    :param target_area: cumulative area we want the data to add up to
    :return: (dict) a dictionary of generated real estate data, and the number of iterations required
    """
    num_entries = len(the_df)  # Number of rows/entries in df

    # Initialise empty dictionary for generated real estate data
    the_dict = dict()

    # Initialise cumulative area
    cumulative_area = 0

    # Loop counter
    jj = 0

    # Continue until the cumulative area of new dictionary is greater than the desired area size
    while cumulative_area < target_area:
        # Generate random integer for indexing the dataframe
        rand_index = np.random.randint(0, num_entries - 1)
        # Get the row from the dataframe
        data_info = the_df.loc[rand_index]

        # Add list of [area, price, area/unit price]
        the_dict[jj] = [data_info['Area'], data_info['Price'], data_info['Area per unit cost']]

        # Update cumulative area
        cumulative_area += data_info['Area']

        # Update counter
        jj += 1
        # Print the counter every 100 iterations
        if jj % 100 == 0:
            print('Loop iteration ', jj)

    print(jj, ' iterations required.')  # Print the total iterations taken

    return the_dict, jj, cumulative_area


# --------- MAIN PROGRAM ---------
# TZS to USD conversion rate in 2022. Source: xe.com
tzs_to_usd_2022 = 0.000428894

# USD 2022 to USD 2012 conversion rate.
usd_2022_to_2012 = 0.81

# Hectare/acre conversion to km^2
hectare_to_km2 = 0.01

acre_to_km2 = 0.0040468564224

scrape = False
if scrape:
    # Main Tanzania url for scraping
    realestate_url = 'https://kupatana.com/tz/search/land-plots'

    # Pages to scrape
    pages = range(2, 3)

    # Initialise dictionary to store data
    data_dict = {}

    # List of urls to scrape
    for page in pages:
        scrape_page(realestate_url, data_dict, page)

    # Save dictionary of real estate data as a dataframe
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Price', 'Currency'])

    file_name = r'\tanzania_data_2.csv'

    df.to_csv(r'H:\Elephant_project\Code' + file_name)

# --- CONVERTING DATA FROM SCRAPED WEBSITE -----
sort_scrape = False

if sort_scrape:
    new_df = pd.read_csv(r'Z:\Elephant_project\Code\tanzania_data_1.csv')

    # Convert all area values in csv file to km2
    new_df = convert_area(new_df, hectare_to_km2, acre_to_km2)

    new_df = convert_price(new_df, tzs_to_usd_2022, usd_2022_to_2012)

    new_df = new_df.sort_values('Area per unit cost', ascending=False)

    # Compute the cumulative area and costs, add these columns to dataframe
    new_df['Area_cumsum'] = new_df['Area'].cumsum()
    new_df['Price_cumsum'] = new_df['Price'].cumsum()

    plt.scatter(np.array(new_df['Price_cumsum']), np.array(new_df['Area_cumsum']))

    file_name = r'\tanzania_data_sort.csv'

    new_df.to_csv(r'Z:\Elephant_project\Code' + file_name)

# ---------- GENERATE NEW FAKE DATA ---------
# Set the random seed
np.random.seed(42)

read_df = pd.read_csv(r'H:\Elephant_project\Code\tanzania_data_sort.csv')

# Target area size - Serenget National Park (km2) Source: Mduma, Sinclair, Hillborn (1999)
serengeti_size = 14763

# Generate dictionary of generated data from existing real data
# as well as number of iterations required
big_data, num_it, tot_area = generate_data(read_df, serengeti_size)

# Add point for (0,0)
big_data[num_it] = [0, 0, 0]

# Sort by area/unit cost
# Sort the data by area/unit price, largest to smallest
# x is the value. x[0] = area,  x[1] = cost, x[-1] = area/unit price
big_sorted_cost = {k: v for k, v in enumerate(sorted(big_data.values(), key=lambda x: x[-1], reverse=True))}

# Convert dictionary to dataframe
big_df = pd.DataFrame.from_dict(big_sorted_cost, orient='index', columns=['Area', 'Price', 'Area per unit cost'])

# Add columns for cumulative sums of area and price
big_df['Area_cumsum'] = big_df['Area'].cumsum()
big_df['Price_cumsum'] = big_df['Price'].cumsum()

file_name_duplicate = r'\tanzania_data_sort_duplicated.csv'

big_df.to_csv(r'H:\Elephant_project\Code' + file_name_duplicate)

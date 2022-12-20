"""Author: Liam Timms 44368768
Created: 06/08/2021
Modified: 13/09/2022
This program is for scraping data from Zambia real estate websites
v1: scrapes Zambian real estate website
v6: create dataframe of duplicated real estate data by taking uniform random draws from the real real estate data set
"""

import re
import requests as rq

from bs4 import BeautifulSoup, Tag
import pandas as pd

import numpy as np


def extract_price_currency(info_list, real_conversion, website_conversion):
    """
    :param info_list: list of the form [amount of area, area units, price string, other stuff etc.]
    :param real_conversion: the real conversion rate 1 ZMK in 2012 to USD
    :param website_conversion: what the website gives as the conversion rate 1 ZMK in 2012 to USD
    :return: info_list: new list of the form [amount of area, area units, price, currency]
    """

    # If 'is ' is at the beginning of the string, cut it off
    if 'is ' in info_list[2]:
        info_list[2] = info_list[2][info_list[2].index(' ') + 1:]

    # Extract the price if there is one
    digits = [char.isdigit() for char in info_list[2]]  # Location of first digit
    if True in digits:
        first_num = [char.isdigit() for char in info_list[2]].index(True)  # Location of first digit

        # Extract the currency
        currency = info_list[2][:first_num]

    # If there is no price information, ditch this ad and move on
    else:
        return

    # Variable for if the price is in millions (e.g. 'K2m' or 'K15m'. One or two numbers followed by 'm')
    million_check = re.search(r'\d{1,2}[m]', info_list[2])

    # If the price string is a 'per hectare' price, cut off the 'per hectare' and denomination, and multiply price by
    # no. of hectares
    if 'per hectare' in ad_info[2] or 'per hectatre' in ad_info[2] or 'per Hectare' in ad_info[2]:
        # Cut off 'per hectare'
        info_list[2] = info_list[2][:info_list[2].index(' ')]

        unit_price = float(info_list[2][first_num:])  # Price per hectare
        tot_price = unit_price * info_list[0]  # Total price

    # Else if the price is in millions then convert to numerical price
    elif million_check:
        # Get the 'K15m' string
        tot_price = million_check.group()
        # Convert to a number
        tot_price = float(tot_price[:-1]) * 1e6

    else:
        # If there are extra words after the price, cut them off
        if ' ' in info_list[2]:
            info_list[2] = info_list[2][:info_list[2].index(' ')]

        # If we can convert info_list[2] to a float, i.e. it is a price, then make it a float
        if True in [char.isdigit() for char in info_list[2]]:
            tot_price = float(info_list[2][first_num:])

        # Otherwise, if there is no price information, ditch this ad and move on
        else:
            return

    # If price is in KWM, convert to USD
    if currency == 'K':
        tot_price = tot_price * real_conversion
        currency = 'US$'
    # If the price is in website $USD, then convert to ZMK (2012) and back to real $USD
    else:
        tot_price = tot_price / website_conversion * real_conversion

    info_list[2] = tot_price

    # If there are 3 or fewer items in list, append currency. Otherwise replace thing[3] with currency
    if len(info_list) < 3.5:
        info_list.append(currency)
    else:
        info_list[3] = currency

    return info_list


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
        the_dict[jj] = [data_info['Area'], data_info['Price'], data_info['Area_per_unit_cost']]

        # Update cumulative area
        cumulative_area += data_info['Area']

        # Update counter
        jj += 1
        # Print the counter every 100 iterations
        if jj % 100 == 0:
            print('Loop iteration ', jj)

    print(jj, ' iterations required.')  # Print the total iterations taken

    return the_dict, jj


# ---------- SCRAPE DATA FROM REAL ESTATE WEBSITE ---------
scrape = False

if scrape:
    # Central Estates Zambia scraping
    realestate_url = 'https://centralestates.org/farms_for_sale'

    # ZMK to $USD conversion rate in 2012. Source: xe.com
    xe_exchange = 0.0002024

    website_exchange = 1 / 18.23739191

    # Hectare/acre conversion to km^2
    hectare_to_km2 = 0.01

    acre_to_km2 = 0.0040468564224

    # Request html data from website
    page = rq.get(realestate_url)

    # Parse contents with BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')

    # Get each advertisement from the soup
    span = soup.find_all('span')

    # Start and end of ads:
    start_index = 19
    end_index = len(span) + 1 - 37

    # Initialise dictionary to store data
    data_dict = {}

    # Split into individual ads
    # For all spans
    for ind, s in zip(range(start_index, end_index), span[start_index:end_index]):
        # For each individual ad

        # Store amount of area, area units, cost, currency, e.g. [10, 'Hectares', 25000, 'K']
        ad_info = []

        # Each ad has 2 sections: amount of land, and cost/details
        for ad in s:
            # If the sentence is a Tag object, print the text as a string
            if type(ad) == Tag:
                area = [str(text) for text in ad][0]

                # Find indices of spaces
                space_index = [0] + [i.start() for i in re.finditer(' ', area)]

                for ii in range(len(space_index) - 1):
                    # Get 1st two substrings: numerical area and units, e.g. area[0:2] = 10 and area[2:11] = 'Hectares'
                    info_segment = area[space_index[ii]:space_index[ii + 1]]

                    # If the string info_segment is not empty
                    if len(info_segment) > 0.1:
                        # If it's a number, convert to float
                        if info_segment[0].isdigit():
                            info_segment = info_segment.replace(',', '')  # Remove any commas
                            info_segment = float(info_segment)

                        else:
                            # Get rid of spaces
                            info_segment = info_segment.strip()

                    # Add the info to area_info list
                    ad_info.append(info_segment)

            # Otherwise, convert to string and print
            else:
                cost_info = ad

                # Ad structure is always "going for K100,000" or "Price/price K100,000"
                # If ad doesn't have "going for ", then try finding "Price "

                # Index of 'going for ', 'price', 'Price', and 'For detatils'
                going_index = cost_info.find('going for ')
                Going_index = cost_info.find('Going for ')
                goingfor_index = cost_info.find('goingfor ')
                price_index = cost_info.find('price')
                Price_index = cost_info.find('Price')
                information_index = cost_info.find('For more')

                details_index = cost_info.find('For det')

                if going_index > 0:
                    cost_info = cost_info[going_index + len('going for '): details_index]
                elif Going_index > 0:
                    cost_info = cost_info[Going_index + len('Going for '): details_index]
                elif goingfor_index > 0:
                    cost_info = cost_info[goingfor_index + len('goingfor '): details_index]
                elif price_index > 0:
                    cost_info = cost_info[price_index + len('price'): details_index]
                elif Price_index > 0:
                    cost_info = cost_info[Price_index + len('Price'): details_index]
                elif information_index > 0:
                    cost_info = cost_info[information_index + len('For more'): details_index]
                else:
                    cost_info = None

                # If the ad has a cost
                if cost_info is not None:
                    # Sometimes 'For more information...' slips through. Find these parts and cut off the tail
                    if cost_info.find('For more') > 0:
                        cost_info = cost_info[:cost_info.find('For more')]

                    elif cost_info.find('For details') > 0:
                        cost_info = cost_info[:cost_info.find('For details')]

                    # If the cost = 'is K1000 per hectare', cut off the 'is '
                    if 'is ' in cost_info:
                        cost_info = cost_info[cost_info.index(' ') + 1:]

                    # If part of the cost_info is a periods that is not a decimal places, then remove it from cost_info.
                    # Keep doing this until all non-decimal periods are gone
                    period_index = cost_info.find('.')
                    while any([c == '.' for c in cost_info[period_index:]]) and period_index != -1:
                        # Cut off end periods
                        if period_index == len(cost_info) - 1:
                            cost_info = cost_info[:-1]
                        # Else if character after period is not a decimal place, cut out the period
                        elif not cost_info[period_index + 1].isdigit():
                            cost_info = cost_info[:period_index] + cost_info[period_index + 1:]

                        # This period is safe or removed, look at tail of string
                        period_index = cost_info.find('.', period_index + 1)

                    # Remove commas
                    cost_info = cost_info.strip().replace(',', '')

                ad_info.append(cost_info)

        # After land amount and units, if there are list elements that are None or are not a price, cut them out
        for t in ad_info[2:len(ad_info) + 1]:
            if (t is None) or (not any([char.isdigit() for char in t])):
                ad_info.remove(t)

        # If the ad data has a cost (i.e. there are no None entries and at least 3 entries), and if this line
        # is not a <span> formatting line, then add it to the data dictionary
        if all(ad_info) and (len(ad_info) > 2.1) and not isinstance(ad_info[0], str):
            ad_info = extract_price_currency(ad_info, xe_exchange, website_exchange)

            # If there is ad data
            if ad_info is not None:
                # If the info contains this font size specifier, remove it
                if '8pt;' in ad_info:
                    ad_info.remove('8pt;')

                # If part of the ad_info is a mobile number (i.e. has a + in it), then remove it from ad_info.
                # Keep doing this until all mobile numbers are gone
                while any([any([p == '+' for p in s]) for s in ad_info if type(s) == str]):
                    for info in ad_info:
                        if type(info) == str:
                            if any([p == '+' for p in info]):
                                ad_info.remove(info)

                # Convert the areas from hectares/acres to km^2
                if ('hect' in ad_info[1].lower()) or ('heat' in ad_info[1].lower()):
                    ad_info[0] = ad_info[0] * hectare_to_km2
                    ad_info[1] = 'km^2'

                elif 'acre' in ad_info[1].lower():
                    ad_info[0] = ad_info[0] * acre_to_km2
                    ad_info[1] = 'km^2'

                data_dict[ind] = ad_info

    # Idea: get cumsum area and price (not adjusted), then sort by area/unit price.
    # i.e. get the regular cumsum area and price, but purchasing cheapest land first.

    # Append area/unit price to dictionary entries
    # Additionally, relabel the keys so they count 0, 1, ...
    for k in data_dict:
        data_dict[k].append(data_dict[k][0] / data_dict[k][2])

    # Sort the data by area/unit price, largest to smallest
    # x is the value. x[0] = area,  x[1] = cost, x[-1] = area/unit price
    sorted_cost = {k: v for k, v in enumerate(sorted(data_dict.values(), key=lambda x: x[-1], reverse=True))}

    # Add row for 0 area = 0 cost
    sorted_cost[0] = [0, 'km^2', 0, 'US$']

    df = pd.DataFrame.from_dict(sorted_cost, orient='index', columns=['Area', 'Area units', 'Price',
                                                                      'Currency', 'Area per unit cost'])

    # Add columns for cumulative sums of area and price
    df['Area_cumsum'] = df['Area'].cumsum()
    df['Price_cumsum'] = df['Price'].cumsum()

    file_name = '\zambia_data_sorted2.csv'

    df.to_csv(r'C:\Users\admin\Documents\University\Honours_project\Code' + file_name)

# ---------- GENERATE NEW FAKE DATA ---------
# Set the random seed
np.random.seed(42)
# This then takes 128822 iterations
# With 0th entry in new_data = [7.34, 742.808, 0.0098814229249011]

# Read in dataframe
read_df = pd.read_csv(r'Z:\Elephant_project\Code\zambia_data_sorted2.csv')

# Target area size
area_size = 753000  # Size of Zambia (km2)
# area_size = 40000   # Size of LVNP (km2)

# Generate dictionary of generated data from existing real data
# as well as number of iterations required
new_data, num_it = generate_data(read_df, area_size)

# Add point for (0,0)
new_data[num_it] = [0, 0, 0]

# Overwrite new_data dictionary with new_data sorted by area/unit cost
# Sort the data by area/unit price, largest to smallest
# x is the value. x[0] = area,  x[1] = cost, x[-1] = area/unit price
new_data = {k: v for k, v in enumerate(sorted(new_data.values(), key=lambda x: x[-1], reverse=True))}

new_df = pd.DataFrame.from_dict(new_data, orient='index', columns=['Area', 'Price', 'Area per unit cost'])

# Add columns for cumulative sums of area and price
new_df['Area_cumsum'] = new_df['Area'].cumsum()
new_df['Price_cumsum'] = new_df['Price'].cumsum()

truncated_df = new_df[::10]     # Slice of df that takes every 10th element


file_name_duplicate = '\zambia_data_sorted_duplicated.csv'

truncated_df.to_csv(r'Z:\Elephant_project\Code' + file_name_duplicate)
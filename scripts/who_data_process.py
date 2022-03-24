import argparse, os, pickle, epiweeks, re, string
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.special import softmax

who_harm_dict = {
    '1.1_Individual measures__Performing hand hygiene': 'Individual measures',
    '1.3_Individual measures__Performing respiratory etiquette': 'Individual measures',
    '1.4_Individual measures__Wearing a mask': 'Masks',
    '1.5_Individual measures__Using other personal protective equipment': 'Individual measures',
    '1.6_Individual measures__Physical distancing': 'Individual measures',
    '2.1_Environmental measures__Cleaning and disinfecting surfaces and objects': 'Environmental measures',
    '2.2_Environmental measures__Improving air ventilation': 'Environmental measures',
    '3.1.1_Surveillance and response measures_Detecting and isolating cases_Passive case detection': 'Detecting and isolating cases',
    '3.1.2_Surveillance and response measures_Detecting and isolating cases_Active case detection': 'Detecting and isolating cases',
    '3.1.3_Surveillance and response measures_Detecting and isolating cases_Isolation': 'Detecting and isolating cases',
    '3.2.1_Surveillance and response measures_Tracing and quarantining contacts_Contact tracing': 'Tracing and quarantining contacts',
    '3.2.2_Surveillance and response measures_Tracing and quarantining contacts_Quarantine of contacts': 'Tracing and quarantining contacts',
    '4.1.1_Social and physical distancing measures_School measures_Adapting': 'School measures',
    '4.1.2_Social and physical distancing measures_School measures_Closing': 'School measures',
    '4.2.1_Social and physical distancing measures_Offices, businesses, institutions and operations_Adapting': 'Offices, businesses, institutions and operations',
    '4.2.2_Social and physical distancing measures_Offices, businesses, institutions and operations_Closing': 'Offices, businesses, institutions and operations',
    '4.3.1_Social and physical distancing measures_Gatherings, businesses and services_Restricting private gatherings at home': 'Gatherings, businesses and services',
    '4.3.2_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, restricting or adapting private gatherings outside the home': 'Gatherings, businesses and services',
    '4.3.3_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, closing, restricting or adapting public gatherings outside the home': 'Gatherings, businesses and services',
    '4.3.4_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, restricting or adapting mass gatherings': 'Gatherings, businesses and services',
    '4.4.1_Social and physical distancing measures_Special populations_Shielding vulnerable groups': 'Special populations',
    '4.4.2_Social and physical distancing measures_Special populations_Protecting populations in closed settings': 'Special populations',
    '4.4.3_Social and physical distancing measures_Special populations_Protecting displaced populations': 'Special populations',
    '4.5.1_Social and physical distancing measures_Domestic travel_Suspending or restricting movement': 'Domestic Travel',
    '4.5.2_Social and physical distancing measures_Domestic travel_Stay-at-home order': 'Stay-at-home order',
    '4.5.3_Social and physical distancing measures_Domestic travel_Restricting entry': 'Domestic Travel',
    '4.5.4_Social and physical distancing measures_Domestic travel_Closing internal land borders': 'Domestic Travel',
    '5.1_International travel measures__Providing travel advice or warning': 'International travel measures',
    '5.2_International travel measures__Restricting visas': 'International travel measures',
    '5.3_International travel measures__Restricting entry': 'International travel measures',
    '5.4_International travel measures__Restricting exit': 'International travel measures',
    '5.5_International travel measures__Entry screening and isolation or quarantine': 'International travel measures',
    '5.6_International travel measures__Exit screening and isolation or quarantine': 'International travel measures',
    '5.7_International travel measures__Suspending or restricting international flights': 'International travel measures',
    '5.8_International travel measures__Suspending or restricting international ferries or ships': 'International travel measures',
    '5.9_International travel measures__Closing international land borders': 'International travel measures',
    '6.1_Drug-based measures__Using medications for prevention': 'Other measures',
    '6.2_Drug-based measures__Using medications for treatment': 'Other measures',
    '8.1_Other measures__Legal and policy regulations': 'Other measures',
    '8.2_Other measures__Scaling up': 'Other measures',
    '8.3_Other measures__Financial packages': 'Financial packages',
    '8.4_Other measures_Communications and engagement_': 'Communications and engagement',
    '8.4.1_Other measures_Communications and engagement_General public awareness campaigns': 'Communications and engagement',
    '8.4.2_Other measures_Communications and engagement_Other communications': 'Communications and engagement',
    '8.5_Other measures__Other': 'Other measures'
 }

label_maps = {
    "who": {'2.1_Environmental measures__Cleaning and disinfecting surfaces and objects': 0,
                '1.5_Individual measures__Using other personal protective equipment': 1,
                '1.4_Individual measures__Wearing a mask': 2,
                '5.9_International travel measures__Closing international land borders': 3,
                '5.5_International travel measures__Entry screening and isolation or quarantine': 4,
                '5.1_International travel measures__Providing travel advice or warning': 5,
                '5.3_International travel measures__Restricting entry': 6,
                '5.7_International travel measures__Suspending or restricting international flights': 7,
                '8.1_Other measures__Legal and policy regulations': 8,
                '4.5.2_Social and physical distancing measures_Domestic travel_Stay-at-home order': 9,
                '4.3.4_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, restricting or adapting mass gatherings': 10,
                '4.2.1_Social and physical distancing measures_Offices, businesses, institutions and operations_Adapting': 11,
                '4.1.2_Social and physical distancing measures_School measures_Closing': 12,
                '4.4.2_Social and physical distancing measures_Special populations_Protecting populations in closed settings': 13,
                '3.1.2_Surveillance and response measures_Detecting and isolating cases_Active case detection': 14,
                '3.1.1_Surveillance and response measures_Detecting and isolating cases_Passive case detection': 15,
                '3.2.1_Surveillance and response measures_Tracing and quarantining contacts_Contact tracing': 16,
                '3.2.2_Surveillance and response measures_Tracing and quarantining contacts_Quarantine of contacts': 17,
                '4.2.2_Social and physical distancing measures_Offices, businesses, institutions and operations_Closing': 18,
                '4.5.1_Social and physical distancing measures_Domestic travel_Suspending or restricting movement': 19,
                '4.3.3_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, closing, restricting or adapting public gatherings outside the home': 20,
                '5.8_International travel measures__Suspending or restricting international ferries or ships': 21,
                '8.4.1_Other measures_Communications and engagement_General public awareness campaigns': 22,
                '4.1.1_Social and physical distancing measures_School measures_Adapting': 23,
                '4.5.3_Social and physical distancing measures_Domestic travel_Restricting entry': 24,
                '3.1.3_Surveillance and response measures_Detecting and isolating cases_Isolation': 25,
                '5.2_International travel measures__Restricting visas': 26,
                '4.3.2_Social and physical distancing measures_Gatherings, businesses and services_Cancelling, restricting or adapting private gatherings outside the home': 27,
                '4.4.3_Social and physical distancing measures_Special populations_Protecting displaced populations': 28,
                '4.4.1_Social and physical distancing measures_Special populations_Shielding vulnerable groups': 29,
                '1.6_Individual measures__Physical distancing': 30,
                '4.5.4_Social and physical distancing measures_Domestic travel_Closing internal land borders': 31,
                '5.4_International travel measures__Restricting exit': 32,
                '4.3.1_Social and physical distancing measures_Gatherings, businesses and services_Restricting private gatherings at home': 33,
                '5.6_International travel measures__Exit screening and isolation or quarantine': 34,
                '6.2_Drug-based measures__Using medications for treatment': 35,
                '6.1_Drug-based measures__Using medications for prevention': 36,
                '1.3_Individual measures__Performing respiratory etiquette': 37,
                '1.1_Individual measures__Performing hand hygiene': 38,
                '2.2_Environmental measures__Improving air ventilation': 39,
                '8.2_Other measures__Scaling up': 40,
                '8.3_Other measures__Financial packages': 41,
                '8.5_Other measures__Other': 42,
                '8.4.2_Other measures_Communications and engagement_Other communications': 43,
                '8.4_Other measures_Communications and engagement_': 44},
    "who_harm": {
        'Individual measures': 6,
        'Masks': 8,
        'Environmental measures': 3,
        'Detecting and isolating cases': 1,
        'Tracing and quarantining contacts': 14,
        'School measures': 11,
        'Offices, businesses, institutions and operations': 9,
        'Gatherings, businesses and services': 5,
        'Special populations': 12,
        'Domestic Travel': 2,
        'Stay-at-home order': 13,
        'International travel measures': 7,
        'Other measures': 10,
        'Financial packages': 4,
        'Communications and engagement': 0
 }}

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and merge WHO documents")
    parser.add_argument("--who_data_path", required=True, help="path to the WHO documents data file")
    parser.add_argument("--aylien_dir", required=True, help="directory containing preprocessed AYLIEN data, including mappings of time and source, as well as the vocab")
    parser.add_argument("--eta_path", required=True, help="path to the inferred eta file, which sould be in the outputs of running MixMedia on AYLIEN")
    parser.add_argument("--save_dir", required=True, help="directory to save outputs")
    
    return parser.parse_args()    

def load_and_process_csv(csv_path, harmonize=True):
    data_df = pd.read_csv(csv_path).drop_duplicates()
    # cleaning
    data_df = data_df.dropna(subset=["SUMMARY", "SOURCE", "DATE ADDED"])
    data_df = data_df[data_df.SUMMARY.apply(lambda text: text.lower()) != 'none']
    data_df['SOURCE'] = data_df['SOURCE'].apply(lambda x: x.strip(" "))
    data_df['SOURCE'] = data_df['SOURCE'].apply(lambda x: x.lower())

    # harmonize
    if harmonize:
        data_df['WHO_MEASURE'] = data_df['WHO_MEASURE'].apply(who_harm_dict.get)

    # encode labels
    label_to_idx = label_maps['who_harm'] if harmonize else label_maps['who']
    n_labels = len(label_to_idx)
    data_df["WHO_MEASURE_ENCODED"] = \
        data_df["WHO_MEASURE"].apply(label_to_idx.get).astype(str)
    
    return data_df

def normalize_time(date_start):
    d = datetime.strptime(date_start, "%d/%m/%Y")

    # code below returns cdc week
    cdc_week = epiweeks.Week.fromdate(d)
    return f"{cdc_week.year}-{cdc_week.week}"

def merge_index_by_summary(who_df, summary, which_index):
    assert which_index in ["time", "source"]
    idxs = []
    for _, row in who_df[who_df["SUMMARY"] == summary].iterrows():
        idxs.append(row["SOURCE INDEX"] if which_index == "source" else row["TIME INDEX"])
    return "|".join(idxs)

def no_punctuation(w):
    return not any(char in string.punctuation for char in w)

def no_numeric(w):
    return not any(char.isdigit() for char in w)

def to_bow(sentence, word_to_idx):
    bow = [0] * len(word_to_idx)
    for word in set(sentence.split()):
        if word in word_to_idx:
            bow[word_to_idx[word]] = 1
    return np.array(bow)

if __name__ == "__main__":
    args = parse_args()

    who_df = load_and_process_csv(args.who_data_path)
    print("WHO data read")

    with open(os.path.join(args.aylien_dir, "sources_map.pkl"), "rb") as file:
        sources_map = pickle.load(file)
    with open(os.path.join(args.aylien_dir, "times_map.pkl"), "rb") as file:
        times_map = pickle.load(file)
    print("Mappings read")

    time_to_idx = {value: key for key, value in times_map.items()}

    def get_time_idx(date_start):
        return time_to_idx.get(normalize_time(date_start))

    who_df["TIME INDEX"] = who_df["DATE ADDED"].apply(get_time_idx)
    who_df["SOURCE INDEX"] = who_df["SOURCE"].apply(sources_map.get)
    who_df = who_df.dropna(subset=["TIME INDEX", "SOURCE INDEX"])
    who_df["TIME INDEX"] = who_df["TIME INDEX"].astype(int).astype(str)
    who_df["SOURCE INDEX"] = who_df["SOURCE INDEX"].astype(int).astype(str)
    print("Source and time indices mapped")

    # merge
    merge_who_df = who_df[["SUMMARY", "WHO_MEASURE_ENCODED"]].groupby("SUMMARY", as_index=False).agg({"WHO_MEASURE_ENCODED": lambda measures: "|".join(measures)})
    merge_who_df["SOURCE INDEX"] = merge_who_df.SUMMARY.apply(lambda summary: merge_index_by_summary(who_df, summary, "source"))
    merge_who_df["TIME INDEX"] = merge_who_df.SUMMARY.apply(lambda summary: merge_index_by_summary(who_df, summary, "time"))

    merge_who_df["MEASURE_CNT"] = merge_who_df["WHO_MEASURE_ENCODED"].apply(lambda measure_str: len(measure_str.split('|')))

    cleaned_summary_tokens = merge_who_df["SUMMARY"].apply(lambda summary: re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', summary))

    cleaned_summary_tokens = cleaned_summary_tokens.apply(lambda tokens: list(filter(no_punctuation, tokens)))
    cleaned_summary_tokens = cleaned_summary_tokens.apply(lambda tokens: list(filter(no_numeric, tokens)))
    cleaned_summary_tokens = cleaned_summary_tokens.apply(lambda tokens: list(filter(lambda token: len(token) > 1, tokens)))

    merge_who_df["SUMMARY"] = cleaned_summary_tokens.apply(lambda tokens: " ".join(tokens))

    output_who_name = os.path.splitext(os.path.basename(args.who_data_path))[0] + "_merged.csv"
    merge_who_df.to_csv(os.path.join(args.save_dir, output_who_name), index=False)
    print("Merged WHO data savaed")

    # bow
    with open(os.path.join(args.aylien_dir, "vocab.pkl"), "rb") as file:
        aylien_vocab = pickle.load(file)
    word_to_aylien_idx = {word: idx for idx, word in enumerate(aylien_vocab)}
    merge_who_bow = np.vstack(merge_who_df.SUMMARY.apply(lambda summary: to_bow(summary, word_to_aylien_idx))).astype(float)
    for idx, bow in enumerate(merge_who_bow):
        if bow.sum() == 0:
            continue
        merge_who_bow[idx] = bow / bow.sum()
    merge_who_bow = sparse.coo_matrix(merge_who_bow)
    sparse.save_npz(os.path.join(args.save_dir, "bow.npz"), merge_who_bow)
    print("BOW saved")

    # eta
    eta = np.load(args.eta_path)
    eta = softmax(eta, axis=-1)

    merge_etas = []

    for _, row in merge_who_df.iterrows():
        source_idxs = [int(source_idx) for source_idx in row["SOURCE INDEX"].split('|')]
        time_idxs = [int(time_idx) for time_idx in row["TIME INDEX"].split('|')]
        assert len(source_idxs) == len(time_idxs)
        
        current_etas = []
        for idx, source_idx in enumerate(source_idxs):
            current_etas.append(eta[source_idx, time_idxs[idx]])
        
        merge_etas.append(np.mean(current_etas, axis=0))
    merge_etas = np.vstack(merge_etas)
    np.save(os.path.join(args.save_dir, "merge_etas.npy"), merge_etas)
    print("Merged etas saved")
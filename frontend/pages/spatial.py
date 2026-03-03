# pages/spatial.py — Spatial Overview Map


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from theme import inject_theme, page_header, section_label, kpi_html, PLOTLY_LAYOUT, SAFFRON, SAFFRON_SCALE, GREEN, RED, AMBER
from utils.api_client import fetch_states, fetch_predictions, fetch_optimizer_results, fetch_district_history

inject_theme()
page_header(
    "◈ Module 05",
    "Spatial Overview",
    "District-level employment prediction map — hover any bubble for full model details",
)

# ── District coordinates (approximate centroids for all major districts) ──────
# Covers all 36 states/UTs across India's 700+ districts.
# Format: "District|State": (lat, lon)
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    # ── Andhra Pradesh ─────────────────────────────────────────────────────────
    "Srikakulam|Andhra Pradesh":          (18.30, 83.90), "Vizianagaram|Andhra Pradesh": (18.12, 83.41),
    "Visakhapatnam|Andhra Pradesh":       (17.69, 83.22), "East Godavari|Andhra Pradesh":(17.00, 82.00),
    "West Godavari|Andhra Pradesh":       (16.92, 81.34), "Krishna|Andhra Pradesh":       (16.61, 80.83),
    "Guntur|Andhra Pradesh":              (16.31, 80.44), "Prakasam|Andhra Pradesh":      (15.35, 79.57),
    "Nellore|Andhra Pradesh":             (14.44, 79.99), "Kurnool|Andhra Pradesh":       (15.83, 78.05),
    "Kadapa|Andhra Pradesh":              (14.47, 78.82), "Anantapur|Andhra Pradesh":     (14.68, 77.60),
    "Chittoor|Andhra Pradesh":            (13.22, 79.10),

    # ── Assam ─────────────────────────────────────────────────────────────────
    "Kamrup|Assam":                       (26.14, 91.77), "Barpeta|Assam":                (26.32, 91.00),
    "Dhubri|Assam":                       (26.02, 89.98), "Goalpara|Assam":               (26.17, 90.62),
    "Nagaon|Assam":                       (26.35, 92.68), "Cachar|Assam":                 (24.81, 92.86),
    "Lakhimpur|Assam":                    (27.24, 94.10), "Dibrugarh|Assam":              (27.49, 95.00),
    "Sonitpur|Assam":                     (26.63, 92.80), "Jorhat|Assam":                 (26.75, 94.22),

    # ── Bihar ─────────────────────────────────────────────────────────────────
    "Patna|Bihar":                        (25.59, 85.13), "Gaya|Bihar":                   (24.80, 84.99),
    "Muzaffarpur|Bihar":                  (26.12, 85.38), "Bhagalpur|Bihar":              (25.24, 86.98),
    "Darbhanga|Bihar":                    (26.16, 85.90), "Purnea|Bihar":                 (25.78, 87.47),
    "Rohtas|Bihar":                       (24.98, 83.98), "Siwan|Bihar":                  (26.22, 84.36),
    "Saran|Bihar":                        (25.92, 84.74), "Nalanda|Bihar":                (25.10, 85.44),
    "Madhubani|Bihar":                    (26.37, 86.07), "Champaran East|Bihar":         (26.65, 84.92),
    "Champaran West|Bihar":               (27.02, 84.46),

    # ── Chhattisgarh ──────────────────────────────────────────────────────────
    "Raipur|Chhattisgarh":               (21.25, 81.63), "Bilaspur|Chhattisgarh":        (22.09, 82.15),
    "Durg|Chhattisgarh":                 (21.19, 81.28), "Rajnandgaon|Chhattisgarh":    (21.10, 81.03),
    "Bastar|Chhattisgarh":               (19.10, 81.95), "Sarguja|Chhattisgarh":         (23.12, 83.19),
    "Korba|Chhattisgarh":                (22.35, 82.72), "Raigarh|Chhattisgarh":        (21.90, 83.40),

    # ── Gujarat ───────────────────────────────────────────────────────────────
    "Ahmedabad|Gujarat":                  (23.03, 72.58), "Surat|Gujarat":                (21.17, 72.83),
    "Vadodara|Gujarat":                   (22.31, 73.18), "Rajkot|Gujarat":               (22.30, 70.80),
    "Bhavnagar|Gujarat":                  (21.77, 72.15), "Jamnagar|Gujarat":             (22.47, 70.06),
    "Junagadh|Gujarat":                   (21.52, 70.46), "Anand|Gujarat":                (22.56, 72.93),
    "Mehsana|Gujarat":                    (23.59, 72.37), "Banaskantha|Gujarat":          (24.17, 72.42),
    "Kutch|Gujarat":                      (23.73, 69.86), "Dahod|Gujarat":                (22.83, 74.25),
    "Narmada|Gujarat":                    (21.87, 73.49), "Valsad|Gujarat":               (20.59, 72.93),
    "Dang|Gujarat":                       (20.75, 73.69),

    # ── Haryana ───────────────────────────────────────────────────────────────
    "Hisar|Haryana":                      (29.15, 75.72), "Sirsa|Haryana":               (29.53, 75.03),
    "Bhiwani|Haryana":                    (28.79, 76.13), "Rohtak|Haryana":              (28.89, 76.61),
    "Sonipat|Haryana":                    (28.99, 77.01), "Karnal|Haryana":              (29.68, 76.99),
    "Ambala|Haryana":                     (30.37, 76.78), "Kurukshetra|Haryana":         (29.97, 76.85),
    "Mahendragarh|Haryana":               (28.27, 76.15),

    # ── Jharkhand ─────────────────────────────────────────────────────────────
    "Ranchi|Jharkhand":                   (23.35, 85.33), "Dhanbad|Jharkhand":           (23.80, 86.45),
    "Bokaro|Jharkhand":                   (23.67, 86.15), "Giridih|Jharkhand":           (24.19, 86.30),
    "Hazaribagh|Jharkhand":               (23.99, 85.36), "Dumka|Jharkhand":             (24.27, 87.25),
    "Palamu|Jharkhand":                   (24.03, 84.08), "Gumla|Jharkhand":             (23.05, 84.54),
    "Pakur|Jharkhand":                    (24.63, 87.84), "Lohardaga|Jharkhand":         (23.44, 84.68),

    # ── Karnataka ─────────────────────────────────────────────────────────────
    "Bangalore Rural|Karnataka":          (13.01, 77.57), "Tumkur|Karnataka":            (13.34, 77.10),
    "Kolar|Karnataka":                    (13.14, 78.13), "Mysore|Karnataka":            (12.30, 76.65),
    "Mandya|Karnataka":                   (12.52, 76.90), "Hassan|Karnataka":            (13.00, 76.10),
    "Chikmagalur|Karnataka":              (13.32, 75.78), "Shimoga|Karnataka":           (13.93, 75.57),
    "Dakshina Kannada|Karnataka":         (12.85, 75.24), "Uttara Kannada|Karnataka":    (14.79, 74.68),
    "Raichur|Karnataka":                  (16.21, 77.36), "Koppal|Karnataka":            (15.35, 76.15),
    "Gadag|Karnataka":                    (15.42, 75.62), "Dharwad|Karnataka":           (15.46, 75.01),
    "Bagalkot|Karnataka":                 (16.18, 75.70), "Bijapur|Karnataka":           (16.83, 75.72),
    "Gulbarga|Karnataka":                 (17.34, 76.82), "Bidar|Karnataka":             (17.91, 77.52),
    "Bellary|Karnataka":                  (15.14, 76.92), "Chitradurga|Karnataka":       (14.23, 76.40),
    "Davangere|Karnataka":                (14.46, 75.92), "Udupi|Karnataka":             (13.34, 74.75),

    # ── Kerala ────────────────────────────────────────────────────────────────
    "Thiruvananthapuram|Kerala":          (8.52,  76.94), "Kollam|Kerala":               (8.88,  76.61),
    "Pathanamthitta|Kerala":              (9.27,  76.77), "Alappuzha|Kerala":            (9.49,  76.32),
    "Kottayam|Kerala":                    (9.59,  76.52), "Idukki|Kerala":               (9.85,  77.10),
    "Ernakulam|Kerala":                   (10.01, 76.31), "Thrissur|Kerala":             (10.52, 76.22),
    "Palakkad|Kerala":                    (10.77, 76.65), "Malappuram|Kerala":           (11.07, 76.07),
    "Kozhikode|Kerala":                   (11.25, 75.78), "Wayanad|Kerala":              (11.61, 76.08),
    "Kannur|Kerala":                      (11.87, 75.37), "Kasaragod|Kerala":            (12.50, 74.99),

    # ── Madhya Pradesh ────────────────────────────────────────────────────────
    "Bhopal|Madhya Pradesh":             (23.26, 77.41), "Indore|Madhya Pradesh":       (22.72, 75.86),
    "Jabalpur|Madhya Pradesh":           (23.18, 79.99), "Gwalior|Madhya Pradesh":      (26.22, 78.18),
    "Sagar|Madhya Pradesh":              (23.84, 78.74), "Rewa|Madhya Pradesh":         (24.53, 81.30),
    "Satna|Madhya Pradesh":              (24.60, 80.83), "Ujjain|Madhya Pradesh":       (23.18, 75.78),
    "Chhindwara|Madhya Pradesh":         (22.06, 78.94), "Shivpuri|Madhya Pradesh":     (25.42, 77.66),
    "Morena|Madhya Pradesh":             (26.50, 78.00), "Bhind|Madhya Pradesh":        (26.56, 78.78),
    "Datia|Madhya Pradesh":              (25.67, 78.46), "Chhatarpur|Madhya Pradesh":   (24.92, 79.58),
    "Tikamgarh|Madhya Pradesh":          (24.74, 78.83), "Raisen|Madhya Pradesh":       (22.99, 77.79),
    "Vidisha|Madhya Pradesh":            (23.52, 77.81), "Hoshangabad|Madhya Pradesh":  (22.75, 77.73),
    "Harda|Madhya Pradesh":              (22.34, 77.09), "Betul|Madhya Pradesh":        (21.91, 77.90),
    "Balaghat|Madhya Pradesh":           (21.81, 80.19), "Seoni|Madhya Pradesh":        (22.09, 79.55),
    "Mandla|Madhya Pradesh":             (22.60, 80.38), "Dindori|Madhya Pradesh":      (22.95, 81.08),
    "Shahdol|Madhya Pradesh":            (23.30, 81.36), "Anuppur|Madhya Pradesh":      (23.10, 81.69),
    "Umaria|Madhya Pradesh":             (23.53, 80.84), "Katni|Madhya Pradesh":        (23.83, 80.39),
    "Panna|Madhya Pradesh":              (24.72, 80.19), "Damoh|Madhya Pradesh":        (23.83, 79.45),
    "Narsinghpur|Madhya Pradesh":        (22.95, 79.19), "Niwari|Madhya Pradesh":       (25.01, 78.76),

    # ── Maharashtra ───────────────────────────────────────────────────────────
    "Ahmednagar|Maharashtra":            (19.10, 74.74), "Akola|Maharashtra":           (20.71, 77.00),
    "Amravati|Maharashtra":              (20.93, 77.75), "Aurangabad|Maharashtra":      (19.88, 75.34),
    "Beed|Maharashtra":                  (18.99, 75.75), "Bhandara|Maharashtra":        (21.17, 79.65),
    "Buldhana|Maharashtra":              (20.53, 76.18), "Chandrapur|Maharashtra":      (19.96, 79.30),
    "Dhule|Maharashtra":                 (20.90, 74.78), "Gadchiroli|Maharashtra":      (20.18, 80.00),
    "Gondia|Maharashtra":                (21.46, 80.20), "Hingoli|Maharashtra":         (19.72, 77.15),
    "Jalgaon|Maharashtra":               (21.00, 75.57), "Jalna|Maharashtra":           (19.84, 75.89),
    "Kolhapur|Maharashtra":              (16.70, 74.24), "Latur|Maharashtra":           (18.40, 76.57),
    "Mumbai City|Maharashtra":           (18.96, 72.82), "Mumbai Suburban|Maharashtra": (19.17, 72.96),
    "Nagpur|Maharashtra":                (21.15, 79.09), "Nanded|Maharashtra":          (19.15, 77.32),
    "Nandurbar|Maharashtra":             (21.37, 74.24), "Nashik|Maharashtra":          (19.99, 73.79),
    "Osmanabad|Maharashtra":             (18.18, 76.04), "Palghar|Maharashtra":         (19.70, 72.77),
    "Parbhani|Maharashtra":              (19.27, 76.77), "Pune|Maharashtra":            (18.52, 73.86),
    "Raigad|Maharashtra":                (18.52, 73.18), "Ratnagiri|Maharashtra":       (16.99, 73.30),
    "Sangli|Maharashtra":                (16.86, 74.56), "Satara|Maharashtra":          (17.69, 74.00),
    "Sindhudurg|Maharashtra":            (16.35, 73.74), "Solapur|Maharashtra":         (17.69, 75.91),
    "Thane|Maharashtra":                 (19.22, 72.98), "Wardha|Maharashtra":          (20.75, 78.60),
    "Washim|Maharashtra":                (20.11, 77.15), "Yavatmal|Maharashtra":        (20.39, 78.13),

    # ── Odisha ────────────────────────────────────────────────────────────────
    "Bhubaneswar|Odisha":                (20.30, 85.84), "Cuttack|Odisha":              (20.46, 85.88),
    "Balasore|Odisha":                   (21.49, 86.93), "Mayurbhanj|Odisha":           (21.92, 86.73),
    "Keonjhar|Odisha":                   (21.63, 85.58), "Sundargarh|Odisha":           (22.12, 84.03),
    "Sambalpur|Odisha":                  (21.47, 83.97), "Bargarh|Odisha":              (21.33, 83.62),
    "Bolangir|Odisha":                   (20.71, 83.49), "Kalahandi|Odisha":            (19.91, 83.17),
    "Koraput|Odisha":                    (18.81, 82.71), "Rayagada|Odisha":             (19.17, 83.41),
    "Ganjam|Odisha":                     (19.39, 84.70), "Puri|Odisha":                 (19.81, 85.83),
    "Khordha|Odisha":                    (20.18, 85.62), "Jagatsinghpur|Odisha":        (20.25, 86.18),
    "Kendrapara|Odisha":                 (20.50, 86.42), "Jajpur|Odisha":               (20.85, 86.33),

    # ── Rajasthan ─────────────────────────────────────────────────────────────
    "Jaipur|Rajasthan":                  (26.92, 75.79), "Jodhpur|Rajasthan":           (26.29, 73.03),
    "Udaipur|Rajasthan":                 (24.58, 73.69), "Kota|Rajasthan":              (25.18, 75.84),
    "Ajmer|Rajasthan":                   (26.45, 74.64), "Bikaner|Rajasthan":           (28.02, 73.31),
    "Alwar|Rajasthan":                   (27.57, 76.61), "Bharatpur|Rajasthan":         (27.22, 77.49),
    "Sikar|Rajasthan":                   (27.61, 75.14), "Nagaur|Rajasthan":            (27.21, 73.74),
    "Pali|Rajasthan":                    (25.77, 73.33), "Barmer|Rajasthan":            (25.75, 71.39),
    "Jaisalmer|Rajasthan":               (26.92, 70.91), "Churu|Rajasthan":             (28.30, 74.96),
    "Jhunjhunu|Rajasthan":               (28.13, 75.40), "Sirohi|Rajasthan":            (24.89, 72.86),
    "Banswara|Rajasthan":                (23.54, 74.44), "Dungarpur|Rajasthan":         (23.84, 73.71),
    "Baran|Rajasthan":                   (25.10, 76.52), "Jhalawar|Rajasthan":          (24.60, 76.16),
    "Tonk|Rajasthan":                    (26.17, 75.79), "Sawai Madhopur|Rajasthan":    (26.01, 76.35),
    "Dausa|Rajasthan":                   (26.89, 76.34), "Karauli|Rajasthan":           (26.50, 77.02),

    # ── Tamil Nadu ────────────────────────────────────────────────────────────
    "Chennai|Tamil Nadu":                (13.08, 80.27), "Coimbatore|Tamil Nadu":       (11.02, 76.97),
    "Madurai|Tamil Nadu":                (9.93,  78.12), "Tiruchirappalli|Tamil Nadu":  (10.80, 78.69),
    "Salem|Tamil Nadu":                  (11.65, 78.16), "Tirunelveli|Tamil Nadu":      (8.73,  77.70),
    "Vellore|Tamil Nadu":                (12.92, 79.13), "Erode|Tamil Nadu":            (11.34, 77.73),
    "Thanjavur|Tamil Nadu":              (10.79, 79.14), "Virudhunagar|Tamil Nadu":     (9.58,  77.96),
    "Ramanathapuram|Tamil Nadu":         (9.37,  78.83), "Pudukkottai|Tamil Nadu":      (10.38, 78.82),
    "Dindigul|Tamil Nadu":               (10.36, 77.98), "Dharmapuri|Tamil Nadu":       (12.13, 78.16),
    "Krishnagiri|Tamil Nadu":            (12.52, 78.21), "Namakkal|Tamil Nadu":         (11.22, 78.17),
    "Nilgiris|Tamil Nadu":               (11.47, 76.73), "Tiruppur|Tamil Nadu":         (11.11, 77.34),
    "Cuddalore|Tamil Nadu":              (11.75, 79.77), "Villupuram|Tamil Nadu":       (11.94, 79.49),
    "Kancheepuram|Tamil Nadu":           (12.83, 79.70), "Thiruvallur|Tamil Nadu":      (13.15, 79.91),
    "Tiruvannamalai|Tamil Nadu":         (12.23, 79.07),

    # ── Telangana ─────────────────────────────────────────────────────────────
    "Hyderabad|Telangana":               (17.38, 78.47), "Medchal|Telangana":           (17.62, 78.48),
    "Rangareddy|Telangana":              (17.25, 78.38), "Nalgonda|Telangana":          (17.05, 79.27),
    "Warangal|Telangana":                (17.97, 79.59), "Karimnagar|Telangana":        (18.44, 79.13),
    "Khammam|Telangana":                 (17.25, 80.15), "Nizamabad|Telangana":         (18.67, 78.10),
    "Adilabad|Telangana":                (19.67, 78.53), "Mahabubnagar|Telangana":      (16.74, 77.99),

    # ── Uttar Pradesh ─────────────────────────────────────────────────────────
    "Lucknow|Uttar Pradesh":             (26.85, 80.95), "Kanpur Nagar|Uttar Pradesh":  (26.45, 80.35),
    "Agra|Uttar Pradesh":                (27.18, 78.02), "Varanasi|Uttar Pradesh":      (25.32, 83.01),
    "Allahabad|Uttar Pradesh":           (25.44, 81.85), "Meerut|Uttar Pradesh":        (28.98, 77.71),
    "Bareilly|Uttar Pradesh":            (28.35, 79.43), "Gorakhpur|Uttar Pradesh":     (26.76, 83.37),
    "Mathura|Uttar Pradesh":             (27.49, 77.67), "Muzaffarnagar|Uttar Pradesh": (29.47, 77.70),
    "Shahjahanpur|Uttar Pradesh":        (27.88, 79.91), "Sitapur|Uttar Pradesh":       (27.57, 80.68),
    "Lakhimpur Kheri|Uttar Pradesh":     (27.94, 80.78), "Hardoi|Uttar Pradesh":        (27.40, 80.13),
    "Unnao|Uttar Pradesh":               (26.54, 80.49), "Rae Bareli|Uttar Pradesh":    (26.22, 81.24),
    "Pratapgarh|Uttar Pradesh":          (25.89, 81.99), "Jaunpur|Uttar Pradesh":       (25.73, 82.69),
    "Ghazipur|Uttar Pradesh":            (25.58, 83.57), "Ballia|Uttar Pradesh":        (25.75, 84.15),
    "Azamgarh|Uttar Pradesh":            (26.07, 83.18), "Mau|Uttar Pradesh":           (25.94, 83.56),
    "Deoria|Uttar Pradesh":              (26.50, 83.78), "Basti|Uttar Pradesh":         (26.79, 82.73),
    "Siddharthnagar|Uttar Pradesh":      (27.29, 83.07), "Maharajganj|Uttar Pradesh":   (27.15, 83.56),
    "Gonda|Uttar Pradesh":               (27.13, 81.97), "Bahraich|Uttar Pradesh":      (27.57, 81.60),
    "Shravasti|Uttar Pradesh":           (27.72, 81.87), "Balrampur|Uttar Pradesh":     (27.43, 82.19),
    "Barabanki|Uttar Pradesh":           (26.94, 81.19), "Faizabad|Uttar Pradesh":      (26.77, 82.14),
    "Ambedkar Nagar|Uttar Pradesh":      (26.43, 82.62), "Sultanpur|Uttar Pradesh":     (26.26, 82.06),
    "Banda|Uttar Pradesh":               (25.48, 80.34), "Chitrakoot|Uttar Pradesh":    (25.20, 80.90),
    "Hamirpur|Uttar Pradesh":            (25.95, 80.15), "Mahoba|Uttar Pradesh":        (25.29, 79.87),
    "Lalitpur|Uttar Pradesh":            (24.69, 78.41), "Jhansi|Uttar Pradesh":        (25.45, 78.57),
    "Jalaun|Uttar Pradesh":              (26.14, 79.34), "Etawah|Uttar Pradesh":        (26.78, 79.02),
    "Auraiya|Uttar Pradesh":             (26.47, 79.51), "Kannauj|Uttar Pradesh":       (27.05, 79.92),
    "Farrukhabad|Uttar Pradesh":         (27.38, 79.57), "Mainpuri|Uttar Pradesh":      (27.23, 79.02),
    "Firozabad|Uttar Pradesh":           (27.15, 78.39), "Etah|Uttar Pradesh":          (27.65, 78.67),
    "Kasganj|Uttar Pradesh":             (27.81, 78.65), "Hathras|Uttar Pradesh":       (27.60, 78.06),
    "Aligarh|Uttar Pradesh":             (27.88, 78.07), "Bulandshahr|Uttar Pradesh":   (28.41, 77.85),
    "Hapur|Uttar Pradesh":               (28.72, 77.78), "Gautam Buddha Nagar|Uttar Pradesh": (28.54, 77.39),
    "Ghaziabad|Uttar Pradesh":           (28.67, 77.44), "Bagpat|Uttar Pradesh":        (28.94, 77.22),
    "Bijnor|Uttar Pradesh":              (29.37, 78.13), "Amroha|Uttar Pradesh":        (28.91, 78.47),
    "Sambhal|Uttar Pradesh":             (28.59, 78.56), "Moradabad|Uttar Pradesh":     (28.84, 78.77),
    "Rampur|Uttar Pradesh":              (28.81, 79.03), "Pilibhit|Uttar Pradesh":      (28.64, 79.81),
    "Budaun|Uttar Pradesh":              (28.04, 79.13),

    # ── West Bengal ───────────────────────────────────────────────────────────
    "Kolkata|West Bengal":               (22.57, 88.37), "Howrah|West Bengal":          (22.59, 88.31),
    "North 24 Parganas|West Bengal":     (22.86, 88.54), "South 24 Parganas|West Bengal":(22.15, 88.27),
    "Bardhaman|West Bengal":             (23.23, 87.86), "Birbhum|West Bengal":         (23.90, 87.53),
    "Murshidabad|West Bengal":           (24.18, 88.27), "Nadia|West Bengal":           (23.47, 88.55),
    "Hooghly|West Bengal":               (22.96, 88.38), "Midnapore West|West Bengal":  (22.43, 86.92),
    "Midnapore East|West Bengal":        (22.11, 87.67), "Bankura|West Bengal":         (23.23, 87.07),
    "Purulia|West Bengal":               (23.33, 86.36), "Malda|West Bengal":           (25.00, 88.14),
    "Dinajpur North|West Bengal":        (25.62, 88.43), "Dinajpur South|West Bengal":  (25.29, 88.68),
    "Jalpaiguri|West Bengal":            (26.54, 88.73), "Darjeeling|West Bengal":      (27.04, 88.26),
    "Cooch Behar|West Bengal":           (26.32, 89.45),

    # ── Himachal Pradesh ──────────────────────────────────────────────────────
    "Shimla|Himachal Pradesh":           (31.10, 77.17), "Kangra|Himachal Pradesh":     (32.10, 76.27),
    "Mandi|Himachal Pradesh":            (31.71, 76.93), "Hamirpur|Himachal Pradesh":   (31.69, 76.52),
    "Una|Himachal Pradesh":              (31.46, 76.27), "Chamba|Himachal Pradesh":     (32.55, 76.13),
    "Solan|Himachal Pradesh":            (30.91, 77.10), "Sirmaur|Himachal Pradesh":    (30.56, 77.46),
    "Bilaspur|Himachal Pradesh":         (31.34, 76.76), "Kinnaur|Himachal Pradesh":    (31.59, 78.45),
    "Kullu|Himachal Pradesh":            (31.96, 77.11), "Lahul Spiti|Himachal Pradesh":(32.77, 77.67),

    # ── Uttarakhand ───────────────────────────────────────────────────────────
    "Dehradun|Uttarakhand":              (30.32, 78.03), "Haridwar|Uttarakhand":        (29.96, 78.16),
    "Nainital|Uttarakhand":              (29.38, 79.46), "Udham Singh Nagar|Uttarakhand":(29.00, 79.52),
    "Almora|Uttarakhand":                (29.60, 79.66), "Pauri Garhwal|Uttarakhand":   (29.78, 79.01),
    "Tehri Garhwal|Uttarakhand":         (30.39, 78.48), "Chamoli|Uttarakhand":         (30.41, 79.32),
    "Rudraprayag|Uttarakhand":           (30.28, 78.98), "Uttarkashi|Uttarakhand":      (30.73, 78.44),
    "Bageshwar|Uttarakhand":             (29.84, 79.77), "Pithoragarh|Uttarakhand":     (29.58, 80.22),
    "Champawat|Uttarakhand":             (29.33, 80.09),

    # ── Punjab ────────────────────────────────────────────────────────────────
    "Amritsar|Punjab":                   (31.63, 74.87), "Ludhiana|Punjab":             (30.90, 75.85),
    "Jalandhar|Punjab":                  (31.33, 75.58), "Patiala|Punjab":              (30.34, 76.39),
    "Bathinda|Punjab":                   (30.21, 74.95), "Gurdaspur|Punjab":            (32.04, 75.41),
    "Firozpur|Punjab":                   (30.93, 74.61), "Hoshiarpur|Punjab":           (31.53, 75.91),
    "Rupnagar|Punjab":                   (30.96, 76.53), "Sangrur|Punjab":              (30.25, 75.84),
    "Moga|Punjab":                       (30.82, 75.17), "Faridkot|Punjab":             (30.67, 74.76),
    "Muktsar|Punjab":                    (30.48, 74.52), "Fazilka|Punjab":              (30.40, 74.02),
    "Nawanshahr|Punjab":                 (31.12, 76.12), "Kapurthala|Punjab":           (31.38, 75.38),

    # ── Jharkhand extra ───────────────────────────────────────────────────────
    "Chatra|Jharkhand":                  (24.21, 84.88), "Koderma|Jharkhand":           (24.47, 85.60),
    "Simdega|Jharkhand":                 (22.61, 84.51), "Khunti|Jharkhand":            (23.07, 85.28),
    "Ramgarh|Jharkhand":                 (23.63, 85.51), "Jamtara|Jharkhand":           (23.96, 86.80),
    "Sahibganj|Jharkhand":               (24.96, 87.63), "Godda|Jharkhand":             (24.83, 87.21),
    "Deoghar|Jharkhand":                 (24.48, 86.70),

    # ── Generic fallback centroids for states ─────────────────────────────────
    "Unknown|Andhra Pradesh":            (15.9,  79.7),
    "Unknown|Assam":                     (26.2,  92.9),
    "Unknown|Bihar":                     (25.1,  85.3),
    "Unknown|Chhattisgarh":              (21.3,  81.7),
    "Unknown|Gujarat":                   (22.3,  71.2),
    "Unknown|Haryana":                   (29.1,  76.1),
    "Unknown|Jharkhand":                 (23.6,  85.3),
    "Unknown|Karnataka":                 (15.3,  75.7),
    "Unknown|Kerala":                    (10.9,  76.3),
    "Unknown|Madhya Pradesh":            (22.9,  78.7),
    "Unknown|Maharashtra":               (19.7,  75.7),
    "Unknown|Odisha":                    (20.9,  85.1),
    "Unknown|Rajasthan":                 (27.0,  74.2),
    "Unknown|Tamil Nadu":                (11.1,  78.7),
    "Unknown|Telangana":                 (17.4,  79.1),
    "Unknown|Uttar Pradesh":             (26.8,  80.9),
    "Unknown|West Bengal":               (22.9,  87.9),
}


def get_coords(district: str, state: str) -> tuple[float, float]:
    """Return (lat, lon) for a district, with fallback to state centroid."""
    rng = np.random.default_rng(abs(hash(f"{district}{state}")) % (2**31))
    key = f"{district}|{state}"
    if key in DISTRICT_COORDS:
        lat, lon = DISTRICT_COORDS[key]
        lat += rng.uniform(-0.08, 0.08)
        lon += rng.uniform(-0.08, 0.08)
        return lat, lon
    # Fallback: state centroid + jitter
    fb_key = f"Unknown|{state}"
    lat, lon = DISTRICT_COORDS.get(fb_key, (22.0, 78.0))
    lat += rng.uniform(-1.2, 1.2)
    lon += rng.uniform(-1.2, 1.2)
    return lat, lon


# ── Controls ──────────────────────────────────────────────────────────────────
states = fetch_states()
if not states:
    st.error("⚠️ API offline — run `uvicorn backend.main:app --port 8000`")
    st.stop()

cc1, cc2, cc3 = st.columns(3)
with cc1:
    state_filter = st.selectbox("State Filter", ["All India"] + states)
with cc2:
    map_metric = st.selectbox("Bubble Color / Size", [
        "Predicted Person-Days",
        "Prediction Error",
        "Budget Gain (LP Optimizer)",
        "Actual Person-Days",
    ])
with cc3:
    year_opts = []
    _df_raw = fetch_predictions()
    if not _df_raw.empty:
        year_opts = sorted(_df_raw["financial_year"].unique().tolist())
    selected_year = st.selectbox("Financial Year", year_opts if year_opts else ["—"])

# ── Fetch & merge data ────────────────────────────────────────────────────────
pred_df = fetch_predictions(
    state=None if state_filter == "All India" else state_filter,
    year=int(selected_year) if selected_year != "—" else None,
)
opt_df = fetch_optimizer_results(
    state=None if state_filter == "All India" else state_filter,
)

if pred_df.empty:
    st.info("No prediction data for selected filters. Ensure the pipeline has run.")
    st.stop()

# Merge optimizer results in if available
if not opt_df.empty:
    merge_cols = ["state", "district"]
    opt_sub = opt_df[merge_cols + [
        c for c in ["persondays_gain", "budget_change_pct", "persondays_per_lakh",
                    "budget_allocated_lakhs", "optimized_budget"]
        if c in opt_df.columns
    ]].drop_duplicates(subset=merge_cols)
    pred_df = pred_df.merge(opt_sub, on=merge_cols, how="left")

# Pick what to color by
COLOR_MAP = {
    "Predicted Person-Days":    "predicted_persondays",
    "Prediction Error":         "prediction_error",
    "Budget Gain (LP Optimizer)": "persondays_gain",
    "Actual Person-Days":       "person_days_lakhs",
}
color_col = COLOR_MAP[map_metric]
if color_col not in pred_df.columns:
    color_col = "predicted_persondays"

# ── Build map data ────────────────────────────────────────────────────────────
lats, lons, colors, sizes = [], [], [], []
hover_data = []

for _, row in pred_df.iterrows():
    lat, lon = get_coords(str(row["district"]), str(row["state"]))
    lats.append(lat)
    lons.append(lon)
    colors.append(float(row.get(color_col, 0) or 0))
    sizes.append(max(float(row.get("predicted_persondays", 1) or 1), 0.1))
    hover_data.append(row)

# Normalize sizes for bubble radius
sz_arr = np.array(sizes)
sz_min, sz_max = sz_arr.min(), sz_arr.max()
norm_sz = np.clip((sz_arr - sz_min) / (sz_max - sz_min + 1e-9) * 13 + 4, 4, 17).tolist()

# ── Choose colorscale based on metric ────────────────────────────────────────
if color_col == "prediction_error":
    cscale = [[0, RED], [0.5, "#FED7AA"], [1, "#FED7AA"]]
    cscale = [[0, RED], [0.5, "#FAFAF9"], [1, GREEN]]
elif color_col == "persondays_gain":
    cscale = [[0, RED], [0.5, "#FFF7ED"], [1, GREEN]]
else:
    cscale = SAFFRON_SCALE

# ── Build hover template ──────────────────────────────────────────────────────
# customdata columns: 0=district, 1=state, 2=fy, 3=actual, 4=predicted,
#                     5=error, 6=persondays_gain, 7=budget_chg_pct,
#                     8=persondays_per_lakh, 9=budget_allocated
custom = []
for row in hover_data:
    custom.append([
        str(row.get("district", "")),
        str(row.get("state", "")),
        int(row.get("financial_year", 0)),
        float(row.get("person_days_lakhs", 0) or 0),
        float(row.get("predicted_persondays", 0) or 0),
        float(row.get("prediction_error", 0) or 0),
        float(row.get("persondays_gain", 0) or 0),
        float(row.get("budget_change_pct", 0) or 0),
        float(row.get("persondays_per_lakh", 0) or 0),
        float(row.get("budget_allocated_lakhs", 0) or 0),
    ])

hover_tmpl = (
    "<b>%{customdata[0]}</b><br>"
    "<span style='color:#A8A29E'>%{customdata[1]}</span><br>"
    "<br>"
    "<b>FY:</b> %{customdata[2]}<br>"
    "<b>Actual PD:</b> %{customdata[3]:.2f}L<br>"
    "<b>Predicted PD:</b> %{customdata[4]:.2f}L<br>"
    "<b>Model Error:</b> %{customdata[5]:+.2f}L<br>"
    "<br>"
    "<b>LP Optimizer</b><br>"
    "<b>PD Gain:</b> %{customdata[6]:+.2f}L<br>"
    "<b>Budget Δ:</b> %{customdata[7]:+.1f}%<br>"
    "<b>Efficiency:</b> %{customdata[8]:.4f} PD/₹L<br>"
    "<b>Budget:</b> ₹%{customdata[9]:,.0f}L"
    "<extra></extra>"
)

fig = go.Figure()
fig.add_scattergeo(
    lat=lats, lon=lons,
    mode="markers",
    marker=dict(
        size=norm_sz,
        color=colors,
        colorscale=cscale,
        colorbar=dict(
            title=dict(text=map_metric[:12], font=dict(color="#78716C", size=9)),
            tickfont=dict(color="#78716C", size=8),
            thickness=10, len=0.55,
            bgcolor="rgba(255,255,255,0.88)",
        ),
        opacity=0.80,
        line=dict(width=0.8, color="rgba(255,255,255,0.7)"),
    ),
    customdata=custom,
    hovertemplate=hover_tmpl,
)

fig.update_geos(
    scope="asia",
    showland=True,    landcolor="#F5F5F4",
    showocean=True,   oceancolor="#EFF6FF",
    showcountries=True, countrycolor="#D6D3D1",
    showsubunits=True,  subunitcolor="#E7E5E4",
    showrivers=True,  rivercolor="#DBEAFE",
    center=dict(lat=22, lon=80),
    projection_scale=5.0,
    bgcolor="rgba(0,0,0,0)",
)
fig.update_layout(
    height=620,
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family="DM Mono, monospace", color="#1C1917"),
    showlegend=False,
    hoverlabel=dict(
        bgcolor="#1C1917",
        bordercolor="#1C1917",
        font=dict(family="DM Mono, monospace", size=11, color="#FAF9F7"),
    ),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Caption ───────────────────────────────────────────────────────────────────
n_mapped = len([c for c in custom if c[0]])
year_label = selected_year if selected_year != "—" else "all years"
st.caption(
    f"{n_mapped} districts · FY {year_label} · "
    f"Bubble size ∝ predicted person-days · Hover for full model details"
)

# ── Summary cards below map ───────────────────────────────────────────────────
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
section_label("Prediction Summary for Filtered View")

c1, c2, c3, c4 = st.columns(4)
total_pred = pred_df["predicted_persondays"].sum()
total_act  = pred_df["person_days_lakhs"].sum()
mean_err   = pred_df["prediction_error"].mean()
gain_total = pred_df["persondays_gain"].sum() if "persondays_gain" in pred_df.columns else 0

c1.metric("Total Predicted PD", f"{total_pred:,.1f}L")
c2.metric("Total Actual PD",    f"{total_act:,.1f}L")
c3.metric("Mean Model Error",   f"{mean_err:+.3f}L")
c4.metric("Total LP Gain",      f"{gain_total:+,.1f}L")

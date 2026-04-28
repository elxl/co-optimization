# --- Transit Physical Parameters ---
# C_L_UNIT = 1400             # Passengers per train capacity
S_SET = [2, 3, 4, 5, 6, 12, 20] # Frequency candidates (trains per hour)
BASE_LOAD_FACTOR = 0      # Base transit load factor

# --- AMoD Physical Parameters ---
C_LANE = 700                # Capacity per lane (vehicles/hr)
AMOD_WAIT = 5               # AMoD average pickup time (minutes)

# --- Economic / General Parameters ---
TRANSIT_FARE = 3.0          # Fixed transit fare ($)

# --- Conversion Factors ---
MPH_TO_M_PER_MIN = 1.6 * 1000 / 60  # Conversion for walking speed
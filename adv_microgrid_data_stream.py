# adv_microgrid_data_stream.py

import datetime
import time
import logging
import pytz
from timezonefinder import TimezoneFinder
import json
import sys
import pvlib
import numpy as np
import matplotlib.pyplot as plt
import pybamm
import matplotlib.dates as mdates



# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_configuration(file_path):
    """
    Load the configuration from a JSON file.

    Args:
        file_path (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration data.

    Raises:
        SystemExit: If the file is not found, JSON is invalid, or the configuration is not a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            config_data = json.load(file)
            if isinstance(config_data, dict):
                return config_data
            else:
                raise TypeError("Configuration file is not a dictionary")
    except FileNotFoundError:
        logging.error(f"Configuration file {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the configuration file: {e}")
        sys.exit(1)
    except TypeError as e:
        logging.error(f"Configuration Error: {e}")
        sys.exit(1)


def fetch_weather_data(lat, lon):
    """
    Fetch weather data from Solcast API or use historical data if the API is unavailable.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        tuple: A tuple containing the weather data (dict) and a boolean indicating whether live data was used.
    """
    api_key = 'atsm46nUXRuSQMQQdyRCA1HRLufvcZwZ'
    url = 'https://api.solcast.com.au/world_radiation/estimated_actuals'

    params = {
        'latitude': lat,
        'longitude': lon,
        'format': 'json',
    }

    logging.info(f"Using json file for debugging purposes.")
    with open('dodoma_weather_data.json', 'r') as file:
        return [json.load(file), False]


    # Uncomment the following code to enable live API fetching
    # try:
    #     response = requests.get(url, params=params, auth=(api_key, ''))
    #     response.raise_for_status()  # Check for HTTP errors
    #     data = response.json()
    #     return data
    # except Exception as err:
    #     logging.error(f"An error occurred accessing the API: {err}. \n\n Attempting to read data from dodoma_weather_data.json now.")
    #     try:
    #         with open('dodoma_weather_data.json', 'r') as file:
    #             return json.load(file)

    #     except Exception as err2:
    #         logging.error(f"An error occurred when accessing the json file: {err2}.")
    #         return None



def get_local_time(lat, lon):
    """
    Determine the local time and timezone based on latitude and longitude.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.

    Returns:
        tuple: A tuple containing the local datetime object and the timezone object.
    """
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    if timezone_str is None:
        logging.warning(f"Could not determine the timezone for location ({lat}, {lon}). Using UTC as fallback.")
        timezone_str = 'UTC'
    try:
        local_tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        logging.warning(f"Unknown timezone '{timezone_str}'. Defaulting to UTC.")
        local_tz = pytz.UTC
    now_utc = datetime.datetime.now(pytz.utc)
    now_local = now_utc.astimezone(local_tz)
    return now_local, local_tz


def get_current_load_demand(hour, critical_load):
    """
    Calculate the load demand based on the time of day using a piecewise load profile.

    Args:
        hour (int): Current hour in local time.
        critical_load (float): Base critical load in Watts.

    Returns:
        float: Total load demand in Watts.
    """
    scale = 80          # Used to scale the values up to 30kW load.
    A1 = 150  * scale # Peak 1 - morning
    A2 = 300  * scale # Peak 2 - evening
    B1 = 200  * scale # Minimum load during the early morning
    B2 = 100  * scale # Minimum load during the day [reference to critical load]
    critical_load = 50
    T1 = 4    # 4 AM - when load starts to increase to first peak
    T2 = 8    # 8 AM - load morning peak
    T3 = 12   # Min load time
    T4 = 18   # 6 PM

    # The piecewise load profile:
    if 0 <= hour < T1:
        # Early morning load profile
        load = critical_load + B1 + (-B1 * np.cos(np.pi * (hour - T1) / 15))
    elif T1 <= hour < T2:
        # Smooth increase to morning peak
        load = critical_load + A1 * (1 - 0.5 * np.cos(np.pi * (hour - T1) / (T2 - T1)))
    elif T2 <= hour < T3:
        # Smoothly transition from morning peak to midday minimum
        load = critical_load + B2 + (A1 - B2) * (0.5 + 0.5 * np.cos(np.pi * (hour - T2) / (T3 - T2)))
    elif T3 <= hour:
        # Smooth increase from midday to evening peak
        load = critical_load + B2 + (A2 - B2) * (0.5 - 0.5 * np.cos((np.pi * 2) * (hour - T3) / (24 - T3)))
    return load





def generate_real_time_data(load_config_path):
    """
    Produces real-time microgrid data every minute and exports to a file.

    Args:
        load_config_path (str): Path to the load configuration JSON file.

    Yields:
        dict: Operational data for the EMS system.
    """
    global current_soc_percent

    # Load the configuration (similar to EMS)
    config = load_configuration(load_config_path)

    # Extract the location (latitude and longitude) from the configuration
    location_data = config.get('location', {})
    latitude = location_data.get('latitude', -1.286389)   # Default to Nairobi if not specified
    longitude = location_data.get('longitude', 36.817223)

    solar_panel = {
        'tilt': 0,          # Panel tilt in degrees
        'azimuth': 0,       # 0 degrees is North, 90 degrees is East, and so on
        'P_rated': 325,     # Rated power per panel in Watts
        'gamma': -0.004,    # Temperature coefficient (1/°C)
        'eta_system': 0.9,  # System efficiency
        'NOCT': 45,         # Nominal Operating Cell Temperature in °C
        'T_STC': 25,        # Standard Test Condition temperature in °C
        'G_STC': 1000,      # Standard irradiance in W/m²
        'longitude': longitude,
        'latitude': latitude
    }
    num_batts = 50000
    # Create the DFN battery model
    model = pybamm.lithium_ion.DFN()
    param_values = pybamm.ParameterValues("Chen2020")

    param_values['Nominal cell capacity [A.h]'] = 50
    param_values['Number of electrodes connected in parallel to make a cell'] = (1 * num_batts)
    param_values['Cell cooling surface area [m2]'] = (0.00531 * num_batts)
    param_values['Cell volume [m3]'] = (0.0000242 * num_batts)
    param_values['Current function [A]'] = "[input]"

    # Calculate the base load as the sum of all critical loads from the config
    critical_load = 5000

    # Number of time intervals
    time_length = 150

    weather_tuple = fetch_weather_data(latitude, longitude)
    using_historical_data = not weather_tuple[1]
    if using_historical_data:
        weather_data = weather_tuple[0]

    soc_data = [0.8]
    solar_gen_data = []
    load_data = []
    current_data = []
    time_arr = []
    loads_shed = []

    # Create the simulation object
    sim = pybamm.Simulation(model, parameter_values=param_values)
    sim.solve(t_eval=[1, time_length], initial_soc=soc_data[-1], inputs={"Current function [A]": 0})  
    dt = 15 * 60
    _, local_tz = get_local_time(latitude, longitude)

    for i in range(14975, time_length + 14975):
        if using_historical_data:
            # Rely on historical data if not using API
            now_local = datetime.datetime.fromisoformat(weather_data[i]['period_end'])
        else:
            # Get local time based on the latitude and longitude in the config
            now_local, local_tz = get_local_time(latitude, longitude)

            # Retrieve live weather data
            weather_data = fetch_weather_data(latitude, longitude)[0]


        # Calculate Net Power Output per panel
        out = calculate_P_output(solar_panel, weather_data[i], [now_local, local_tz])
        
        watts_per_panel = out['p_mp']                       # Find the Watt output from each panel per 15 minute intervals
        solar_generation = watts_per_panel.values[0] * 120  # Accumulate total energy for the array for 120 panels

        load_demand = get_current_load_demand(now_local.hour, critical_load)

        # Calculate new SoC based on generation and load
        net_power = round(solar_generation - load_demand, 0)  # Net power in Watts
        net_critical = round(solar_generation - critical_load, 0)

        voltage = sim.solution["Terminal voltage [V]"].data[-1]
        
        ems_power, loads_were_shed = run_ems(net_power, soc_data, current_data, net_critical, sim, critical_load)

        loads_shed.append(loads_were_shed)

        current = (ems_power * -1) / voltage
        current_data.append(current)

        sim.step(dt=dt, inputs={"Current function [A]": current})

        stoichiometry = sim.solution["X-averaged negative particle stoichiometry"].entries  # Multiply by 10 to get the SoC %

        if voltage > 4.19:
            logging.error(f'Voltage has reached dangerously high levels. Battery is assumed to have exploded.')
            break
        elif voltage < 3.51:
            logging.error(f'Voltage has reached dangerously low levels. Battery is damaged beyond repair.')
            break
        elif stoichiometry[-1][-1] < 0.02:
            logging.error(f'----------Battery reached critical failure.----------')
            break

        soc_data.append(stoichiometry[-1][-1])
        solar_gen_data.append(solar_generation)
        load_data.append(load_demand)
        time_arr.append(now_local)

        logging.info(f"t = {i} | Net power: {net_power} W | SOC: {round(soc_data[-1], 2)} | Terminal voltage: {round(sim.solution["Terminal voltage [V]"].data[-1], 3)} V | Current: {round(current, 2)} | Local time: {now_local}")

        # Sleep for 60 seconds to simulate real-time data generation
        if not using_historical_data:
            time.sleep(60)  # 60 seconds for real-time simulation


    # After collecting data, plot it
    plot_data(soc_data, solar_gen_data, load_data, time_arr, loads_shed)
    
def run_ems(net_power, soc_data, current_data, net_critical, sim, critical):
    """
    EMS logic to determine battery charging/discharging and load shedding.

    Args:
        net_power (float): Net power after subtracting load from generation.
        soc_data (list): List of State of Charge (SoC) values.
        current_data (list): List of battery current values.
        net_critical (float): Net power considering only critical load.
        sim (pybamm.Simulation): PyBaMM simulation object.
        critical_load (float): Critical load in Watts.

    Returns:
        tuple: Tuple containing EMS power (float) and a boolean indicating if loads were shed.
    """

    soc = soc_data[-1]
    prev_voltage = sim.solution["Terminal voltage [V]"].data[-1]

    if prev_voltage is None:
        estimated_voltage = 4.08  
    else:
        estimated_voltage = prev_voltage
    current = -net_power / estimated_voltage  

    # Using Ohmic Losses to predict the Terminal Voltage
    ohmic_losses = sim.solution['X-averaged solid phase ohmic losses [V]'].entries

    if len(ohmic_losses) < 2 or len(current_data) < 2:
        ohmic_losses = 0
        ohmic_loss_v = 0
        current_for_ohmic_losses = 0
    else:

        ohmic_loss_v = np.mean(ohmic_losses[-2:-1])
        current_for_ohmic_losses = np.mean(current_data[-2:-1])
   
    if current_for_ohmic_losses == 0:
        predicted_voltage = 3.9
        r_internal_ohmic_losses = -1
        ocv = -1
    else:
        ocv = 3.0 + soc * (4.2 - 3.0)   # Linear approximation between 3.0 V and 4.2 V
        r_internal_ohmic_losses = ohmic_loss_v / current_for_ohmic_losses
        predicted_voltage = ocv - current * r_internal_ohmic_losses + 0.07   #    <---- Decimal is added for discrepancy in calculations


    load = net_critical + critical - net_power

    if predicted_voltage > 4:
        logging.info(f"High Voltage. Halt charging.")
        return [load * -1, True]    
    elif predicted_voltage < 3.65:
        logging.info(f"Critically low voltage. Battery discharge is now {-1 * max(net_critical+critical, 0)}.")
        return [max(net_critical+critical, 0), True]
    elif predicted_voltage < 3.7:
        logging.info("Low Voltage. Shed all non-critical loads.")
        return [net_critical, True]
    elif soc_data[-1] < 0.2:
        logging.info(f"Critically low SoC. Shed all loads.")
        return [0, True]
    elif soc_data[-1] < 0.3:
        logging.info(f"Low SoC. Shed all non-critical loads.")
        return [net_critical, True]
    elif soc_data[-1] > 0.9:
        logging.info(f"High SoC. Slow down power supply.")
        return [net_power / 1.5, True]
    else:
        return [net_power, False]




def plot_data(soc_data, solar_gen_data, load_data, time_arr, loads_shed):
    """
    Plots the solar generation, battery SoC, and load demand over time.

    Args:
        soc_data (list): List of State of Charge (SoC) values.
        solar_gen_data (list): List of solar generation values.
        load_data (list): List of load demand values.
        time_arr (list): List of datetime objects corresponding to each time step.
        loads_shed (list): List of booleans indicating if loads were shed at each time step.

    Returns:
        None
    """
    soc_data = soc_data[:len(time_arr)]
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot solar generation data
    ax1.plot(time_arr, solar_gen_data, marker='o', linestyle='-', color='purple', label='Solar Generation')

    # Plot load data
    ax1.plot(time_arr, load_data, marker='s', linestyle='-', color='orange', label='Load Consumption')

    # Set labels and title for primary y-axis
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power (W)')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M', tz=datetime.timezone.utc))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis

    # Plot SoC data on the secondary y-axis
    ax2.plot(time_arr, soc_data, marker='^', linestyle='-', color='blue', label='SoC')
    ax2.set_ylim(0, 1)

    # Set label for secondary y-axis
    ax2.set_ylabel('State of Charge (%)')


    # Get handles and labels from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Combine them and display
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Highlight times when loads are shed
    end_reached = True
    stop = True
    for i, shed in enumerate(loads_shed):
        if shed and stop:
            end_reached = False
            for j, shed_j in enumerate(loads_shed[i:]):
                if not shed_j:
                    ax1.axvspan(time_arr[i], time_arr[i+j-1], color=(1, 0.4, 0.4, 0.3))
                    end_reached = True
                    break
            stop = False

        if not end_reached:
            if len(loads_shed) > len(time_arr):
                ax1.axvline(x=time_arr[len(time_arr)-1], marker='x', color='red', linestyle='-', linewidth=1.5)
                ax1.axvspan(time_arr[i-1], time_arr[len(time_arr)-1], color=(1, 0.4, 0.4, 0.3))
                break

            ax1.axvspan(time_arr[i], time_arr[len(time_arr)-1], color=(1, 0.4, 0.4, 0.3))
            break

        if not shed:
            stop = True



    plt.title('Solar Generation, Load Consumption, and SoC Over Time')
    ax1.grid(True)

    plt.show()


def calculate_P_output(panel, weather_data, time_data):
    """
    Calculate the net power output of a solar panel array based on weather data and panel characteristics.

    Args:
        panel (dict): Dictionary containing panel specifications and location.
        weather_data (dict): Dictionary containing weather data for the current time step.
        time_data (list): List containing the current datetime object and timezone.

    Returns:
        dict: Output from the single diode model containing power and current characteristics.
    """
    site = pvlib.location.Location(panel['latitude'], panel['longitude'], tz=time_data[1])
    solar_position = site.get_solarposition(times=time_data[0])
    G_POA = pvlib.irradiance.get_total_irradiance(surface_tilt=panel['tilt'], surface_azimuth=panel['azimuth'],
                                                    solar_zenith=solar_position['apparent_zenith'], solar_azimuth=solar_position['azimuth'], 
                                                    dni=weather_data['dni'], ghi=weather_data['ghi'], dhi=weather_data['dhi'], albedo=weather_data['albedo'])
    
    # Calculate cell temperature using the PVsyst model
    temp_cell = pvlib.temperature.pvsyst_cell(poa_global=G_POA['poa_global'], temp_air=weather_data['air_temp'], wind_speed=weather_data['wind_speed_10m'])

    # Module specifications at STC
    i_sc_ref = 9.17                 # Short circuit current [A]
    v_oc_ref = 46.38                # Open circuit voltage [V]
    i_mp_ref = 8.69                 # Current at maximum power point [A]
    v_mp_ref = 37.39                # Voltage at maximum power point [V]
    gamma_pmp_ref = -0.4            # Rated maximum power [%/C]
    cells_in_series = 72            # Number of cells in series
    T_ref = 25                      # Reference temperature [°C]

    # Temperature coefficients (%/°C to per unit/°C)
    alpha_sc = (0.058 / 100) * i_sc_ref  # [A/°C]
    beta_oc = (-0.330 / 100) * v_oc_ref  # [V/°C]


    # Estimate the module parameters using the CEC model fitting function
    params = pvlib.ivtools.sdm.fit_cec_sam(
        v_oc=v_oc_ref,
        i_sc=i_sc_ref,
        v_mp=v_mp_ref,
        i_mp=i_mp_ref,
        alpha_sc=alpha_sc,
        beta_voc=beta_oc,
        cells_in_series=cells_in_series,
        temp_ref=T_ref,
        celltype='polySi',
        gamma_pmp=gamma_pmp_ref
    )

    # Unpack the parameters
    Il_ref, I0_ref, Rs_ref, Rsh_ref, a_ref, Adjust = params

    # Define module parameters dictionary
    module_parameters = {
        'I_L_ref': Il_ref,
        'I_o_ref': I0_ref,
        'R_s': Rs_ref,
        'R_sh_ref': Rsh_ref,
        'a_ref': a_ref,
        'Adjust': Adjust,
        'alpha_sc': alpha_sc,
        'cells_in_series': cells_in_series
    }

    # Environmental conditions
    iam = pvlib.iam.ashrae(solar_position['zenith'])
    effective_irradiance = G_POA.values[0][0] * iam
    temp_cell = 25  # [°C]


    # Calculate the single diode parameters at given conditions
    IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance,
        temp_cell,
        module_parameters['alpha_sc'],
        module_parameters['a_ref'],
        module_parameters['I_L_ref'],
        module_parameters['I_o_ref'],
        module_parameters['R_sh_ref'],
        module_parameters['R_s'],
    )

    # Calculate the IV curve
    out = pvlib.pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)

    return out




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python adv_microgrid_data_stream.py <load_config_path>")
        sys.exit(1)

    # Typically load_config.json
    load_config_path = sys.argv[1]

    try:
        data_generator = generate_real_time_data(load_config_path)
    except KeyboardInterrupt:
        logging.info("Program stopped by user")
        exit(0)
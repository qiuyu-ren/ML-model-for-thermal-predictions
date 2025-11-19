import os
import pandas as pd
import matplotlib.pyplot as plt
import time


time_steps = 10



def process_file(folder_name, file_name, write_folder):
    df = None
    if(".csv" in file_name):
        df = pd.read_csv(f"./{folder_name}/{file_name}")
    else:
        df = pd.read_excel(f"./{folder_name}/{file_name}")

    '''
    df.plot(x="Time (s)", y=["T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)"], title="Temperature vs Time")

    plt.title(f"Old Data: {folder_name} - {file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.show()
    '''
    df.rename(columns={'T_ave (C)': 'T_avg (C)'}, inplace=True)

    times = df['Time (s)'].tolist()
    outer_temps = df['T_outer (C)'].tolist()
    inner_temps = df['T_inner (C)'].tolist()
    avg_temps = df['T_avg (C)'].tolist()
    input_temps = df['Input Temperature (C)'].tolist()

    last_index = 0
    next_index = 1

    rows = []

    for t in range(0, int(times[-1]+1), time_steps):
        while(next_index < len(times)-1 and t >= times[next_index]):
            last_index += 1
            next_index += 1
        
        if(t == times[last_index]):
            rows.append({"Time (s)": times[last_index], "T_outer (C)": outer_temps[last_index], "T_inner (C)": inner_temps[last_index], "T_avg (C)": avg_temps[last_index], "Input Temperature (C)": input_temps[last_index]})
        else:
            ratio = (t - times[last_index]) / (times[next_index] - times[last_index])
            interp_outer = outer_temps[last_index] + ratio * (outer_temps[next_index] - outer_temps[last_index])
            interp_inner = inner_temps[last_index] + ratio * (inner_temps[next_index] - inner_temps[last_index])
            interp_avg = avg_temps[last_index] + ratio * (avg_temps[next_index] - avg_temps[last_index])
            interp_input = input_temps[last_index] + ratio * (input_temps[next_index] - input_temps[last_index])
            rows.append({"Time (s)": t, "T_outer (C)": interp_outer, "T_inner (C)": interp_inner, "T_avg (C)": interp_avg, "Input Temperature (C)": interp_input})
            #print(f"last_index: {times[last_index]}, next_index: {times[next_index]}, Ratio: {ratio:.4f} | Time: {t}s | Outer: {interp_outer:.2f}C | Inner: {interp_inner:.2f}C | Avg: {interp_avg:.2f}C | Input: {interp_input:.2f}C")

    new_df = pd.DataFrame(rows)
    new_df.to_csv(f"../{write_folder}/{folder_name}/{file_name}", index=False)

    #print(new_df.head(96))

    
    new_df.plot(x='Time (s)', y=['T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)'], figsize=(8,5))
    plt.title('Temperature Profiles')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig(f"../{write_folder+'_plot'}/{folder_name}/{file_name.split('.')[0]+'.png'}", dpi=300, bbox_inches='tight')
    plt.close()



def process_folders_and_files(root_dir, write_folder):
    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        '''
        if(folder_name == os.path.basename(root_dir)):
            continue
        '''
        print(f"\n📁 Folder: {folder_name}")
        os.makedirs(f"./{write_folder}/{folder_name}", exist_ok=True)
        for file in files:
            if('.py' in file):
                continue
            print(f"   └── {file}")
            process_file(folder_name, file, write_folder)



if __name__ == "__main__":
    root_directory = "./" 
    write_folder = f"test_in_{time_steps}s"
    os.makedirs(f"../{write_folder}", exist_ok=True)
    os.makedirs(f"../{write_folder+'_plot'}", exist_ok=True)
    process_folders_and_files(root_directory, write_folder)
    #process_file("test_with_inputs", "(test)Case (11).csv", write_folder)

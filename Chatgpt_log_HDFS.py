import openai
import pandas as pd
# Set up API key
openai.api_key = ""

system_intel = """
You need to extract the Content of the logs and replace the dynamic variables with templates. Please output the results directly without explanation.
1. Extraction and Replacement Rules:
   - Time Replacement: Replace all timestamps and specific times with "TIME".
   - PID Replacement: Replace all process IDs (e.g., 081109 203615 148) with "PID".
   - Block ID: Replace the numbers following the block identifier (e.g., 38865049064139660) with "NUM".
   - IP Address Replacement: Replace IP addresses with "IP".
   - File Path Replacement: Replace file paths with "ADDR".
   - Number Replacement: Replace all numbers in log entries with "NUM".
   - User Replacement: Replace all usernames with "USER".
2. Example Log Replacement:
   - Original Log: 081109 214043 2561 dfs.DataNode$DataXceiver: 10.251.30.85:50010:Got exception while serving blk_-2918118818249673980 to /10.251.90.64:
   - Replaced Log: "IP": Got exception while serving blk_"NUM" to /"IP":
Please extract the log template from this log message:
"""

log_file_path = 'HDFS.log'
output_csv_path = 'processed_data_HDFS.csv'
data = []
# Read and process each line of log files
with open(log_file_path, 'r') as file:
    for line_number, line in enumerate(file, 1):
        # Utilize OpenAI's API directly
        try:
            result = openai.ChatCompletion.create(model="gpt-4",
                                                  messages=[{"role": "system", "content": system_intel},
                                                            {"role": "user", "content": line}])
            processed_text = result['choices'][0]['message']['content']
            print(f"Line {line_number}: {processed_text}")
            # Add original and processed data to a list
            data.append({'Original Text': line.strip(), 'Processed Text': processed_text})
        except Exception as e:
            print(f"An error occurred: {e}")

# Create a DataFrame using the collected data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print("Processing complete, data saved: ", output_csv_path)

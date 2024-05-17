import openai
import pandas as pd

# Set up API key
openai.api_key = ""

system_intel = """
You need to extract the Content of the logs and replace the dynamic variables with templates. Please output the results directly without explanation.
1. Extraction and Replacement Rules:
   - Time Replacement: Replace all timestamps and specific times with "TIME".
   - IP Address Replacement: Replace IP addresses with "IP".
   - File Path Replacement: Replace file paths with "ADDR".
   - Number Replacement: Replace all numbers in log entries with "NUM".
   - User Replacement: Replace all usernames with "USER".
2. Example Log Replacement:
   - Original Log: 1134724900 2005.12.16 R43-M1-NC-I:J18-U01 2005-12-16-01.21.40.588712 R43-M1-NC-I:J18-U01 RAS APP FATAL ciod: Error reading message prefix on CioStream socket to 172.16.96.116:39416, Link has been severed
   - Replaced Log: ciod: Error reading message prefix on CioStream socket to "IP", Link has been severed
Please extract the log template from this log message:
"""

log_file_path = 'BGL.log'
output_csv_path = 'processed_data_BGL.csv'
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

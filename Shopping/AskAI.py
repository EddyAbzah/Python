from google import genai

client = genai.Client()


def ask_gemini(query):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query
    )
    return response.text


def print_gemini(query):
    response = ask_gemini(query)
    print(response)


def ask_gemini_motorcycle_data(motorcycle, data_csv="Mileage mpg, MSRP $"):
    query = f"""Return the following information as CSV only:
    {data_csv}
    
    for the following:
    {motorcycle}
    
    No explanations. No extra text."""
    return ask_gemini(query)

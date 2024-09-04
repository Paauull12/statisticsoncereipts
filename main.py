import easyocr
import time
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import json

model_name = "mychen76/invoice-and-receipts_donut_v1"

processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.eval()

def increment_letter(letter):
    if letter == 'Z':
        raise ValueError("Cannot increment beyond 'Z'")
    return chr(ord(letter) + 1)

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.reader = easyocr.Reader(['en'])  # Initialize your EasyOCR reader or other models here

    def get_model(self):
        return self.reader

def get_easyocr_reader():
    model_loader = ModelLoader()
    return model_loader.get_model()

def replace_except_chars(s, chars_to_keep, replacement_char):
    pattern = f'[^{re.escape(chars_to_keep)}]'
    return re.sub(pattern, replacement_char, s)

def getNumber(stringThatIsAlMostAnumber):
    stringThatIsAlMostAnumber = stringThatIsAlMostAnumber.strip()
    stringThatIsAlMostAnumber = stringThatIsAlMostAnumber.replace(',', '.')
    chars_to_keep = "1234567890."
    return replace_except_chars(stringThatIsAlMostAnumber, chars_to_keep, "")

def resultIsTotal(text):
    formattedText = text.strip().lower()
    possibleResult = ["total", "balance", "net total", "payment", "amount"]
    return formattedText in possibleResult

def findTextOnTheSameLineWithC(thetotalword, word, c):
    bbox = np.array(thetotalword, dtype=np.int32)
    firsty = bbox[0][1]
    secondy = bbox[3][1]
    bbox = np.array(word, dtype=np.int32)
    newfirsty = bbox[0][1]
    newsecondy = bbox[3][1]
    return firsty - c <= newfirsty and secondy + c >= newsecondy

def imageProcessorWithEasyocr(img_path):
    start_time = time.time()
    reader = get_easyocr_reader() 
    results = reader.readtext(img_path)

    listOfTotals = [result for result in results if resultIsTotal(result[1])]

    listOfPrices = []
    for result in listOfTotals:
        for words in results:
            if findTextOnTheSameLineWithC(result[0], words[0], 40) and result != words:
                number = getNumber(words[1])
                regex = r'^[^0-9]*([0-9]+\.[0-9]{2}).*$'
                pattern = re.compile(regex)
                
                match = pattern.search(number)

                if match:
                    listOfPrices.append(float(getNumber(match.group(1))))
    end_time = time.time()
    return max(listOfPrices) if listOfPrices else "", end_time - start_time

def getInformationFromReceipt(img_path):
    start_time = time.time()
    try:
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        total_matches = re.findall(r'<s_total>(.*?)</s_total>', output_text)
        
        if total_matches:
            total_text = total_matches[0]
            
            regex = r'^[^0-9]*([0-9]+\.[0-9]{2}).*$'
            pattern = re.compile(regex)
            
            match = pattern.search(total_text)

            end_time = time.time()
            if match:
                return match.group(1), end_time - start_time
        
        return  getNumber(total_matches[0]), end_time - start_time
    except Exception as e:
        print(f"Error processing receipt: {e}")
        return ""

def mainFunctionForStatistics():
    json_file_path = './test/metadata.jsonl'
    workbook = load_workbook('statictisc.xlsx')
    sheet = workbook.active

    final_data = {}
    index = 0
    with open(json_file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())  
                final_data[index] = data
                index += 1
                #pretty_data = json.dumps(data, indent=4) 
                
            except json.JSONDecodeError:
                print("Error decoding a line in the JSONL file")
    
    green_fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")
    red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    yello_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    row = 2
    number_of_runs = 9
    for key in final_data:
        values = final_data[key]

        file_name = values['file_name']
        file_path = './test/' + file_name

        final_price = values['ground_truth']['gt_parse']['total']
        final_price = getNumber(final_price)

        regex = r'^[^0-9]*([0-9]+\.[0-9]{2}).*$'
        pattern = re.compile(regex)
        match = pattern.search(final_price)
        if match:
            final_price = match.group(1)

        result1, time1 = imageProcessorWithEasyocr(file_path)
        result2, time2 = getInformationFromReceipt(file_path)

        column = 'A'

        print("---------------------------------------------------------")
        print(file_path + " -> " + final_price + " and the result are -> " + str(result1) + " in " + str("{:.2f}".format(time1)) + " seconds > " + str(result2)+ " in " + str("{:.2f}".format(time2)) + " seconds ")
        print("---------------------------------------------------------")
        #now that we have the info we need to start to fill the statisctis

        val1 = getNumber(str(result1)).strip()
        val2 = getNumber(str(result2)).strip()
        compareto = getNumber(str(final_price)).strip()
        sheet[column+str(row)].value = file_name

        column = increment_letter(column)
        sheet[column+str(row)].value = getNumber(str(final_price))

        column = increment_letter(column)
        sheet[column+str(row)].value = getNumber(str(result1))

        column = increment_letter(column)
        sheet[column+str(row)].value = "{:.2f}".format(time1)

        column = increment_letter(column)
        if val1 == compareto:
            sheet[column+str(row)].fill = green_fill
        elif val1 in compareto and val1:
            print(f"aici avem un fals >{val1}<")
            sheet[column+str(row)].fill = yello_fill
        else:
            sheet[column+str(row)].fill = red_fill

        column = increment_letter(column)
        sheet[column+str(row)].value = getNumber(str(result2))

        column = increment_letter(column)
        sheet[column+str(row)].value = "{:.2f}".format(time2)

        column = increment_letter(column)
        if val2 == compareto:
            sheet[column+str(row)].fill = green_fill
        elif val2 in compareto and val2:
            print(f"aici avem un fals {val2}")
            sheet[column+str(row)].fill = yello_fill
        else:
            sheet[column+str(row)].fill = red_fill

        row += 1

        if row >= number_of_runs:
            break

    workbook.save('statictisc.xlsx')
    


mainFunctionForStatistics()

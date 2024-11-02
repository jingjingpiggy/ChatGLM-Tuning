import sys
import os
# print(os.getcwd())
sys.path.append(os.path.abspath(os.getcwd()+"/ChatGLM-Tuning"))

from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from cover_alpaca2jsonl import format_example
import json


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


model = AutoModel.from_pretrained("THUDM/chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map='auto')
model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

instructions = json.load(open("/content/ChatGLM-Tuning/data/alpaca_data.json"))


with torch.no_grad():
    for idx, item in enumerate(instructions[:10]):
        feature = format_example(item)
        input_text = feature["context"]
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            temperature=0
        )
        answer = tokenizer.decode(out[0])
        print(answer)
        item['infer_answer'] = answer
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')

###运行2分20秒之后的答案
# Loading checkpoint shards: 100%
#  8/8 [00:13<00:00,  1.51s/it]
# /usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1417: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation )
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
#   warnings.warn(
# /usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1535: UserWarning: You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on cpu, whereas the model is on cuda. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('cuda') before running `.generate()`.
#   warnings.warn(
# WARNING:transformers_modules.THUDM.chatglm-6b.bf0f5cfb575eebebf9b655c5861177acfee03f16.modeling_chatglm:The dtype of attention mask (torch.int64) is not bool
# Instruction: Give three tips for staying healthy.
# Answer: 1. Eat a balanced diet that includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.
# 2. Exercise regularly, either through physical activity or simply by walking or doing light exercises.
# 3. Get enough sleep each night, as sleep is essential for maintaining good health. Aim for 7-9 hours of sleep per night to help your body recover and function properly.
# ### 1.Answer:
#  1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.
# 2. Exercise regularly to keep your body active and strong.
# 3. Get enough sleep and maintain a consistent sleep schedule.
#
#
# Instruction: What are the three primary colors?
# Answer: The three primary colors are red, blue, and yellow. These colors are used in painting, printing, and other visual arts to create colors that can be easily mixed together to create other colors.
# ### 2.Answer:
#  The three primary colors are red, blue, and yellow.
#
#
# Instruction: Describe the structure of an atom.
# Answer: The structure of an atom is made up of a central nucleus containing a small amount of positively charged protons and negatively charged neutrons, surrounded by a cloud of electrons. The electrons are located in specific energy levels around the nucleus, and they are responsible for carrying away the energy from the nucleus and allowing it to be released as light and heat. The electrons are also responsible for making up the atom's chemical properties. The size of an atom is typically measured in terms of the distance between the nucleus and the electrons, known as the electron radius. The total mass of an atom is typically around 1/6 of the mass of the sun.
# ### 3.Answer:
#  An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.
#
#
# Instruction: How can we reduce air pollution?
# Answer: There are several ways to reduce air pollution:
#
# 1. Reduce energy consumption: Reducing energy consumption can help reduce the amount of greenhouse gases that are released into the atmosphere, which contribute to air pollution. This can be achieved by using energy-efficient appliances and lighting, reducing energy use in buildings, and transitioning to renewable energy sources.
# 2. Reduce plastic waste: Plastic waste is a major contributor to air pollution. Reducing plastic waste can help reduce the amount of plastic that is released into the environment, which can contribute to air pollution. This can be achieved by reducing the use of single-use plastics, recycling, and reducing the amount of plastic that is
# ### 4.Answer:
#  There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances.
#
#
# Instruction: Describe a time when you had to make a difficult decision.
# Answer: I had to make a difficult decision when I was faced with the option of continuing my studies or starting a new job. I had been working as a software engineer for several years and had just received a promotion at my current company. However, I was also interested in pursuing a degree in computer science, and I knew that it would be a good way to advance my career.
#
# I was torn between the two options, and I struggled with the decision for several weeks. I felt like I was making a mistake not continuing my studies, but I also knew that starting a new job would be a good opportunity for me to grow and make more money
# ### 5.Answer:
#  I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.
#
#
# ###
# Instruction: Give three tips for staying healthy.
# Answer: 1. Eat a balanced diet that includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.
# 2. Exercise regularly, either through physical activity or simply by walking or doing light exercises.
# 3. Get enough sleep each night, as sleep is essential for maintaining good health. Aim for 7-9 hours of sleep per night to help your body recover and function properly.
# ### 1.Answer:
#  1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.
# 2. Exercise regularly to keep your body active and strong.
# 3. Get enough sleep and maintain a consistent sleep schedule.
#
#
# Instruction: What are the three primary colors?
# Answer: The three primary colors are red, blue, and yellow. These colors are used in painting, printing, and other visual arts to create colors that can be easily mixed together to create other colors.
# ### 2.Answer:
#  The three primary colors are red, blue, and yellow.
#
#
# Instruction: Describe the structure of an atom.
# Answer: The structure of an atom is made up of a central nucleus containing a small amount of positively charged protons and negatively charged neutrons, surrounded by a cloud of electrons. The electrons are located in specific energy levels around the nucleus, and they are responsible for carrying away the energy from the nucleus and allowing it to be released as light and heat. The electrons are also responsible for making up the atom's chemical properties. The size of an atom is typically measured in terms of the distance between the nucleus and the electrons, known as the electron radius. The total mass of an atom is typically around 1/6 of the mass of the sun.
# ### 3.Answer:
#  An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.
#
#
# Instruction: How can we reduce air pollution?
# Answer: There are several ways to reduce air pollution:
#
# 1. Reduce energy consumption: Reducing energy consumption can help reduce the amount of greenhouse gases that are released into the atmosphere, which contribute to air pollution. This can be achieved by using energy-efficient appliances and lighting, reducing energy use in buildings, and transitioning to renewable energy sources.
# 2. Reduce plastic waste: Plastic waste is a major contributor to air pollution. Reducing plastic waste can help reduce the amount of plastic that is released into the environment, which can contribute to air pollution. This can be achieved by reducing the use of single-use plastics, recycling, and reducing the amount of plastic that is
# ### 4.Answer:
#  There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances.
#
#
# Instruction: Describe a time when you had to make a difficult decision.
# Answer: I had to make a difficult decision when I was faced with the option of continuing my studies or starting a new job. I had been working as a software engineer for several years and had just received a promotion at my current company. However, I was also interested in pursuing a degree in computer science, and I knew that it would be a good way to advance my career.
#
# I was torn between the two options, and I struggled with the decision for several weeks. I felt like I was making a mistake not continuing my studies, but I also knew that starting a new job would be a good opportunity for me to grow and make more money
# ### 5.Answer:
#  I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.
#
#
# Instruction: Identify the odd one out.
# Input: Twitter, Instagram, Telegram
# Answer: Telegram is the odd one out, as it is not a traditional social media platform like Twitter or Instagram. It is a messaging and file sharing platform that was founded in 2009 and is currently owned by the company that is behind Facebook, WhatsApp.
# ### 6.Answer:
#  Telegram
#
#
# Instruction: Explain why the following fraction is equivalent to 1/4
# Input: 4/16
# Answer: 4/16 is equivalent to 1/4 because 4 divided by 16 is equal to 1 divided by 4. This can be shown by using the long division method, where the first step is to find the greatest common divisor (GCD) of 4 and 16, which is 4. Then, 4 divided by 4 is equal to 1.
# ### 7.Answer:
#  The fraction 4/16 is equivalent to 1/4 because both numerators and denominators are divisible by 4. Dividing both the top and bottom numbers by 4 yields the fraction 1/4.
#
#
# Instruction: Write a short story in third person narration about a protagonist who has to make an important career decision.
# Answer: As the protagonist sat in her office, staring at the computer screen, she couldn't shake the feeling of uncertainty that had settled in her stomach. She had been working in her field for years, and had never truly felt like she had found her calling.
#
# The decision she had to make was simple, yet difficult. She had to choose between her current job and a new opportunity that offered more growth and potential. She knew that she wanted more, but she also knew that her current job was where she had been comfortable and where she had a strong foundation.
#
# As she thought about it further,
# ### 8.Answer:
#  John was at a crossroads in his life. He had just graduated college and was now facing the big decision of what career to pursue. After much deliberation, he decided that he wanted to be an accountant and help the financially disadvantaged. He had always been good with numbers and enjoyed seeing the tangible results of his work.
#
# John enrolled in accounting courses and initially found it quite challenging. He had to learn multiple systems and regulations quickly, but he worked hard and eventually excelled in his studies. After a few years, John started working at an accounting firm in his city. He was eager to put his knowledge of taxes and accounting to use in a real-world setting.
#
# John loved his job, as it let him express his creativity in finding strategies to save his clients money. After a few years at the firm, he became a senior accountant and was asked to manage bigger and more challenging cases. He was now a respected figure in the financial industry, but he still remembers when he was just a recent college graduate, unsure of the direction in which his life would take him.
#
#
# Instruction: Render a 3D model of a house
# Answer: As an AI language model, I don't have the capability to render 3D models directly. However, there are several online resources that can help you create a 3D model of a house:
#
# 1. 3D modeling software: There are many 3D modeling software programs available that can help you create a 3D model of a house. Some popular options include Blender, Autodesk AutoCAD, and SketchUp. These programs typically require a user to input 2D images or plans of the house and then use various tools to create the 3D model.
# 2. Online 3D modeling websites: There are also
# ### 9.Answer:
#  <nooutput> This type of instruction cannot be fulfilled by a GPT model.
#
#
# Instruction: Evaluate this sentence for spelling and grammar mistakes
# Input: He finnished his meal and left the resturant
# Answer: The sentence is grammatically correct. However, it could be improved by adding more context or information. For example, "He finished his meal and left the restaurant" would be a more complete sentence.
# ### 10.Answer:
#  He finished his meal and left the restaurant.
#
#
# Instruction: Give three tips for staying healthy.
# Answer:
# ### 1.Answer:
#  1.Eat a balanced diet and make sure to include plenty of fruits and vegetables.
# 2. Exercise regularly to keep your body active and strong.
# 3. Get enough sleep and maintain a consistent sleep schedule.
#
#
# Start generating for example 2...
# Instruction: What are the three primary colors?
# Answer:
# ### 2.Answer:
#  The three primary colors are red, blue, and yellow.
#
#
# Start generating for example 3...
# Instruction: Describe the structure of an atom.
# Answer:
# ### 3.Answer:
#  An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.
#
#
# Start generating for example 4...
# Instruction: How can we reduce air pollution?
# Answer:
# ### 4.Answer:
#  There are a number of ways to reduce air pollution, such as shifting to renewable energy sources, encouraging the use of public transportation, prohibiting the burning of fossil fuels, implementing policies to reduce emissions from industrial sources, and implementing vehicle emissions standards. Additionally, individuals can do their part to reduce air pollution by reducing car use, avoiding burning materials such as wood, and changing to energy efficient appliances.
#
#
# Start generating for example 5...
# Instruction: Describe a time when you had to make a difficult decision.
# Answer:
# ### 5.Answer:
#  I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.
#
#
# Start generating for example 6...
# Instruction: Identify the odd one out.
# Input: Twitter, Instagram, Telegram
# Answer:
# ### 6.Answer:
#  Telegram
#
#
# Start generating for example 7...
# Instruction: Explain why the following fraction is equivalent to 1/4
# Input: 4/16
# Answer:
# ### 7.Answer:
#  The fraction 4/16 is equivalent to 1/4 because both numerators and denominators are divisible by 4. Dividing both the top and bottom numbers by 4 yields the fraction 1/4.
#
#
# Start generating for example 8...
# Instruction: Write a short story in third person narration about a protagonist who has to make an important career decision.
# Answer:
# ### 8.Answer:
#  John was at a crossroads in his life. He had just graduated college and was now facing the big decision of what career to pursue. After much deliberation, he decided that he wanted to be an accountant and help the financially disadvantaged. He had always been good with numbers and enjoyed seeing the tangible results of his work.
#
# John enrolled in accounting courses and initially found it quite challenging. He had to learn multiple systems and regulations quickly, but he worked hard and eventually excelled in his studies. After a few years, John started working at an accounting firm in his city. He was eager to put his knowledge of taxes and accounting to use in a real-world setting.
#
# John loved his job, as it let him express his creativity in finding strategies to save his clients money. After a few years at the firm, he became a senior accountant and was asked to manage bigger and more challenging cases. He was now a respected figure in the financial industry, but he still remembers when he was just a recent college graduate, unsure of the direction in which his life would take him.
#
#
# Start generating for example 9...
# Instruction: Render a 3D model of a house
# Answer:
# ### 9.Answer:
#  <nooutput> This type of instruction cannot be fulfilled by a GPT model.
#
#
# Start generating for example 10...
# Instruction: Evaluate this sentence for spelling and grammar mistakes
# Input: He finnished his meal and left the resturant
# Answer:
# ### 10.Answer:
#  He finished his meal and left the restaurant.





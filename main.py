import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deployment import preprocess, detect

# init
device = 'cpu'  # use 'cuda:0' if GPU is available
# model_dir = "nealcly/detection-longformer" # model in our paper
model_dir = "yaful/MAGE"  # model in the online demo
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir).to(device)

messages = [
    """
There are many ways to make coffee, each with its own flavor and texture.
A drip coffee maker is the easiest—just add water, a coffee filter, and
ground coffee (1-2 tablespoons per cup), then let it brew. A French press
gives a rich taste: add coarse-ground coffee, pour hot water (~200°F),
steep for 4 minutes, and press the plunger down. For a strong espresso,
use an espresso machine by tamping finely ground coffee into the portafilter
and extracting a shot in 25-30 seconds. A pour-over (like a V60 or Chemex)
involves adding medium-ground coffee to a filter, pouring hot water in circles,
and letting it drip through. If you need something fast, instant coffee only
requires stirring 1-2 teaspoons into hot water. Each method offers a unique coffee experience!
""",

    """
I make coffee by adding my Keurig cup into the Keurig machine. Placing my coffee cup under the machine. Then pressing one of three options
8oz, 10oz or 12oz. Wait for the coffee machine to fill up my cup, add my creamer and enjoy. I ususally add a lot of creamer in mine
""",

    """
To shoot a basketball, start with a balanced stance,
feet shoulder-width apart. Grip the ball with your shooting hand under
it and your guide hand on the side. Bend your knees, align your elbow,
and focus on the hoop. Push the ball upward using your legs and wrist,
releasing it with a flick of your fingers for backspin. Follow through
by extending your arm and keeping your wrist relaxed. Aim for a soft arc
and release at the peak of your jump. Practice and consistency improve accuracy!
""",

    """
I was taught how to shoot a basketball by following BEEF. 1. Bend your knees. 2. Eyes on the basket
3. Bend Elbow straight to the basket. 4. Follow through
"""
]

index = 0

for msg in messages:
    print('Input:', msg, '\n')

    # preprocess
    text = preprocess(msg)

    # print('preprocssed \n', text, '\n')
    # detection
    result = detect(text, tokenizer, model, device)

    print('Expected:', 'machine' if index % 2 == 0 else 'human')
    print('Predicted', result, '\n')
    index += 1

import argparse
import os
from openai import OpenAI


class TopicLabeller(object):

    def __init__(
        self,
        model,
        temperature=0.2,
        max_tokens=1000,
        frequency_penalty=0.0
    ) -> None:
        try:
            self._openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise Exception(
                "Please set the OPENAI_API_KEY environment variable.")

        self._client = OpenAI(
            api_key=self._openai_api_key
        )

        example_1 = ('network, traffic, vehicle, energy, communication, service, deep, reinforcement, sensor, wireless, road, channel, management, node, UAV',
                     'Traffic Management and Autonomous Driving')

        self.parameters = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of labelling chemical descriptions of the topics of a certain topic model. For example, if I give you the chemical description {example_1[0]}, you will give me the label {example_1[1]}. Just answer with the label, no need to write anything else."
                 },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty
        }

    def set_parameters(self, **kwargs) -> None:
        """Set parameters for the OpenAI model.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to set.
        """

        for key, value in kwargs.items():
            if key != "messages":
                self.parameters[key] = value

    def update_messages(
        self,
        messages: list
    ) -> None:
        """Update the messages of the OpenAI model, always keeping the first message (i.e., the system role)

        Parameters
        ----------
        messages : list
            A list of messages to update the model with.
        """

        self.parameters["messages"] = [
            self.parameters["messages"][0], *messages]

        return

    def _promt(self, gpt_prompt) -> str:
        """Promt the OpenAI ChatCompletion model with a message.

        Parameters
        ----------
        gpt_prompt : str
            A message to promt the model with.

        Returns
        -------
        str
            The response of the OpenAI model.
        """

        message = [{"role": "user", "content": gpt_prompt}]
        self.update_messages(message)
        response = self._client.chat.completions.create(
            **self.parameters
        )
        return response.choices[0].message.content

    def get_label(self, chem_desc: str) -> str:
        """Get a label for a chemical description.

        Parameters
        ----------
        chem_desc : str
            A chemical description.

        Returns
        -------
        str
            A label for the chemical description.
        """

        gpt_prompt = f"Give me a label for this set of words: {chem_desc}"
        return self._promt(gpt_prompt)

    def get_labels(self, chem_descs: list) -> list:
        """Get labels for a list of chemical descriptions.

        Parameters
        ----------
        chem_descs : list
            A list of chemical descriptions.

        Returns
        -------
        list
            A list of labels for the chemical descriptions.
        """

        gpt_prompt = f"Give me a label for each of the following set of words and return it as a Python list with the labels: {chem_descs}"
        return eval(self._promt(gpt_prompt))

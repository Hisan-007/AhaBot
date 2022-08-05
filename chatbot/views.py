from rest_framework.response import Response
from rest_framework.views import APIView
from chatbot.utils.excecutor import get_answer


class TalkToBot(APIView):
    permission_classes = []

    def post(self, request, *args, **kwargs):
        question = request.data["question"]

        answer = get_answer(question)
        data = {
            "answer": answer,
        }
        return Response({'data': data})
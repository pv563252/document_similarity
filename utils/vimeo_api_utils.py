import vimeo
import os


class VimeoAPI:
    def __init__(self):
        self.client = vimeo.VimeoClient(token=os.getenv('VIMEO_SESSION_TOKEN'),
                                        key=os.getenv('VIMEO_CLIENT_KEY'),
                                        secret=os.getenv('VIMEO_CLIENT_SECRET'))

    def get_video_tags(self, clip_id):
        """
        Request Function to get Video Tags
        :param clip_id: Integer Clip ID of the Video
        :return: Response Content
        """
        response = self.client.get('/videos/' + str(clip_id) + '/tags')
        return response.content

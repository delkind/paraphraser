import os
import shutil
from time import gmtime, strftime


class GoogleDrive():
    def __init__(self):
        # !pip install - U - q PyDrive
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials

        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def save_to_drive(self, pathname):
        file_list = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        data_from_colab = None
        for file1 in file_list:
            if file1['title'] == 'data_from_colab':
                # print ('title: %s, id: %s' % (file1['title'], file1['id']))
                data_from_colab = file1['id']

        if (data_from_colab):
            f = self.drive.CreateFile({'parents': [{u'id': data_from_colab}]})
            f.SetContentFile(pathname)
            f.Upload()
            f_id = f['id']
            print(f'saving {pathname} as id {f_id}')
            return f_id
        else:
            raise Exception('did not find')

    def load_from_drive(self, id_in_drive, local_pathname):  # find it from share
        down = self.drive.CreateFile({'id': id_in_drive})  # {'id': '1nIcmbxc6c3Nic35O2JrSEmTmq7X6D2a2'}
        down.GetContentFile(local_pathname)


class Persistency():
    """
    usage:
    Persistency().save_weights(model.get_models(), model.get_model_names(), save_to_gdrive=True)  # not run yet....
    # saving models/models_weights_2018-07-17_19-40-05.zip as id 1kxKYkP0iojqYZ6JY5kNvhdrGSjMTqYMX

    #pers.load_weights_from(model.get_models(), model.get_model_names(), folder='from_gdrive/', download_gdrive_id=load_id)

    """
    def __init__(self):
        self.models_base_dir = 'models'
        if not os.path.exists(self.models_base_dir):
            os.makedirs(self.models_base_dir)

    def save_weights(self, models, model_names, save_to_gdrive=False):

        # keras can save model architecture + weights + opt_state using model.save
        # It can also sepately save weights and architecture as json
        # When using custom code (like our loss function), one need to use CustomObject (TODO)

        # we have 3 top-level-models
        # model = encoder_model + decoder_model
        # d = d_classfier_model + d_encoder_model
        # adv = encoder_model , classfier_model, decoder_model
        # but they share few of their weights
        time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

        models_time_dir = f'{self.models_base_dir}/{time_str}/'
        if not os.path.exists(models_time_dir):
            os.makedirs(models_time_dir)

        # TODO: split part of model, decoder_sampling_model to decoder weights only
        for i in range(len(models)):
            models[i].save_weights(models_time_dir + model_names[i] + '.h5')

        # !ls -lh {self.models_base_dir}
        shutil.make_archive(f'{self.models_base_dir}/models_weights_{time_str}', 'zip', models_time_dir)
        # !ls -lh {self.models_base_dir}

        if (save_to_gdrive):
            drive = GoogleDrive()
            drive.save_to_drive(f'{self.models_base_dir}/models_weights_{time_str}.zip')

    def load_weights_from(self, models, model_names, folder, download_gdrive_id=None):
        """ id_in_drive - default None. If not, download from google-drive (id is the shared-link id)"""
        time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        if download_gdrive_id:
            self.drive = GoogleDrive()
            uploaded_zip = f'{self.models_base_dir}/models_weights_{time_str}_OUT.zip'
            self.drive.load_from_drive(download_gdrive_id, uploaded_zip)
            # extract_dir=f'{models_base_dir}/{time_str}/')
            shutil.unpack_archive(uploaded_zip, folder)
            print('loaded into ', folder)

        for i in range(len(models)):
            pathname = f'{folder}/{model_names[i]}.h5'
            if not os.path.exists(pathname):
                print(pathname, 'missing. breaking!')
                return
            else:
                print('loading', pathname)
            models[i].load_weights(pathname)
        print('load from ', folder, 'complete. Make sure you already compiled the model and set optimizers')





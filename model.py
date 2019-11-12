import os, time, scipy.io
import tensorflow as tf
import numpy as np
from Network import spatio_temporal_encoder,spatio_encoder,integrated_decoder
from utils import *
import scipy
import h5py
import glob
frame_length=4
class denoiser(object):
    def __init__(self, sess, is_color=1, sigma=20, lamda=1.0,scale=2, batch_size=128):
        self.sess = sess
        self.is_color=is_color
        if self.is_color:
           self.input_c_dim = 3
        else:
           self.input_c_dim = 1
        self.sigma = sigma
        self.lamda=lamda
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None,frame_length, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        
        with tf.variable_scope('share_encoder'):#spatio-temporal encoder
             self.share_features = spatio_temporal_encoder(self.X, is_training=self.is_training)
        [self.frame1,self.frame2,self.frame3,self.frame4]=tf.split(self.X,frame_length,axis=1)

        self.frame1=tf.squeeze(self.frame1,axis=1)
        self.frame2=tf.squeeze(self.frame2,axis=1)
        self.frame3=tf.squeeze(self.frame3,axis=1)
        self.frame4=tf.squeeze(self.frame4,axis=1)
        self.concat_frame=tf.concat([self.frame1,self.frame2,self.frame3,self.frame4],axis=0)
        with tf.variable_scope('private_encoder'):#spatio encoder
             self.specific_features =spatio_encoder(self.concat_frame, is_training=self.is_training) 
        [self.specific_features1,self.specific_features2,self.specific_features3,self.specific_features4]=tf.split(self.specific_features,frame_length,axis=0)     

        self.share_features=tf.squeeze(self.share_features,axis=1)

        self.concat_features1=tf.concat([self.specific_features1,self.share_features],axis=3)
        self.concat_features2=tf.concat([self.specific_features2,self.share_features],axis=3)
        self.concat_features3=tf.concat([self.specific_features3,self.share_features],axis=3)
        self.concat_features4=tf.concat([self.specific_features4,self.share_features],axis=3)
        self.concat_features=tf.concat([self.concat_features1, self.concat_features2, self.concat_features3, self.concat_features4], axis=0)

        with tf.variable_scope('share_decoder'):#integrated decoder
             self.noise_frame = integrated_decoder(self.concat_features,  is_training= self.is_training, output_channels=self.input_c_dim)
        #residual learning
        self.Denoise_frame = self.concat_frame-self.noise_frame 
        self.Denoise_frame = tf.expand_dims(self.Denoise_frame,axis=1)
        [self.Denoise_frame1,self.Denoise_frame2,self.Denoise_frame3,self.Denoise_frame4]=tf.split(self.Denoise_frame,frame_length,axis=0)
        self.Y=tf.concat([self.Denoise_frame1,self.Denoise_frame2,self.Denoise_frame3,self.Denoise_frame4],axis=1)
        
        self.loss = (1.0 / (batch_size*frame_length)) * tf.nn.l2_loss(self.Y_ - self.Y)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
             self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    def evaluate(self, iter_num, test_data, sample_dir, summary_merged, summary_writer):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        #lines = open('data/test.list', 'r')
        lines = test_data

        clips = []
        psnr_sum_all=0
        for ii in range(len(lines)):
            line = lines[ii].strip('\n').split()
            dirname = './data/'+ line[0]
            print(dirname)
            for parent, dirnames, filenames in os.walk(dirname):
                filenames = sorted(filenames)
                test_vedio = []
                for i in range(0, len(filenames)):
                    image_name = str(dirname) + '/' + str(filenames[i])
                    if self.is_color:
                        img = Image.open(image_name).convert("RGB")
                    else:
                        img = Image.open(image_name).convert("L")
                    img_data = np.reshape(np.array(img, dtype="uint8"),(img.size[1], img.size[0], self.input_c_dim))
                    test_vedio.append(img_data)
            test_vedio=np.array(test_vedio).astype(np.float32)/255.0
            test_vedio=np.expand_dims(test_vedio,axis=0)
            count=0
            psnr_sum=0
            for idx in range(0,len(filenames)-frame_length+1,frame_length):
                clean_image = test_vedio[:,idx:idx+frame_length,:,:,:]
                output_clean_image,noisy_image,psnr_summary= self.sess.run([self.Y,self.X,summary_merged],
                                   feed_dict={self.Y_: clean_image,                                                   
                                              self.is_training: False})
                #print(np.shape(output_clean_image))
                summary_writer.add_summary(psnr_summary, iter_num)
                groundtruth = np.clip(255*test_vedio[:,idx:idx+frame_length,:,:,:], 0, 255).astype('uint8')
                noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
                outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
                # calculate PSNR
                for frame_num in range(frame_length):

                    psnr = cal_psnr(groundtruth[0,frame_num,:,:,:], outputimage[0,frame_num,:,:,:])
                    print("img%d PSNR: %.2f" % (count + 1, psnr))
                    psnr_sum += psnr
                    count=count+1
                    save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                             groundtruth[0,frame_num,:,:,:], noisyimage[0,frame_num,:,:,:], outputimage[0,frame_num,:,:,:],self.is_color)
            avg_psnr = psnr_sum / count
            psnr_sum_all+=avg_psnr
            print(dirname)
            print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
            f=open("test_delta_5.txt","a+")
            f.write("Epoch: %d , Average PSNR: %.3f---"%(iter_num,avg_psnr)+"\n")
        avg_all=psnr_sum_all/len(lines)
        f.write("Epoch: %d , Average ALL Dataset: %.3f---"%(iter_num,avg_all)+"\n")
        f.write("\n")
        f.close()


    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)
        iter_num = 0
        start_epoch = 0
        start_step = 0
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)  # eval_data value range is 0-255
        for epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            
            for batch_id in range(start_step, numBatch):
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :, :]
                # batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
                _, loss,summary = self.sess.run([ self.train_op,self.loss ,merged],
                                        feed_dict={self.Y_: batch_images,   #
                                                   self.lr: lr[epoch],
                                                   self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch+1, eval_every_epoch) == 0:
            #if np.mod(iter_num, 2500) == 0:
                self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                                summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='M2MNet'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    #NOTE: train with batch size 
    def test(self, test_path, paras, save_dir):

        print("[*] restore model...")
        saver = tf.train.Saver()
        saver.restore(self.sess, paras)
        print("[*] Load weights success...")
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")

        psnr_sum = 0
        test_data=glob.glob(os.path.join(test_path,'*.jpg'))
        filenames=sorted(test_data)
        assert len(filenames) != 0, 'No testing data!'
        test_vedio = []
        for i in range(0, len(filenames)):
            image_name = filenames[i]
            if self.is_color:
               img = Image.open(image_name).convert("RGB")
            else:
               img = Image.open(image_name).convert("L")
            img_data = np.reshape(np.array(img, dtype="uint8"),(img.size[1], img.size[0], self.input_c_dim))
            test_vedio.append(img_data)
        test_vedio=np.array(test_vedio).astype(np.float32)/255.0
        test_vedio=np.expand_dims(test_vedio,axis=0)
        count=0
        psnr_sum=0
        mse=0

        for idx in range(0,len(filenames)-frame_length+1,frame_length):
             clean_image = test_vedio[:,idx:idx+frame_length,:,:,:]
             start_time = time.time()

             output_clean_image,noisy_image= self.sess.run([self.Y,self.X],
                                   feed_dict={self.Y_: clean_image,                                                   
                                              self.is_training: False})
             average_frame_time=(time.time()-start_time)/frame_length
             #print(np.shape(output_clean_image)
             groundtruth = np.clip(255*test_vedio[:,idx:idx+frame_length,:,:,:], 0, 255).astype('uint8')
             noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
             outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
             # calculate PSNR
             for frame_num in range(frame_length):
                 mse_frame = ((groundtruth.astype(np.float) - outputimage.astype(np.float)) ** 2).mean()
                 mse =mse + mse_frame
                 psnr = cal_psnr(groundtruth[0,frame_num,:,:,:], outputimage[0,frame_num,:,:,:])
                 print("img%d PSNR: %.2f, test time: %.2f" % (count + 1, psnr,average_frame_time))
                 psnr_sum += psnr
                 count=count+1
                 save_images(os.path.join(save_dir, 'test%d.png' % count),outputimage[0,frame_num,:,:,:], self.is_color)
        avg_psnr = psnr_sum / count
        mse=mse/count
        sequence_psnr = 10 * np.log10(255 ** 2 / mse)
        print("--- Test ---- Average frame PSNR %.2f ---" % avg_psnr)
        print("--- Test ---- sequence PSNR %.2f ---" % sequence_psnr)




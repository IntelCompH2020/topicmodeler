"""
*** IntelComp H2020 project ***
*** Topic Modeling Toolbox  ***

Provides two main classes for Topic Modeling
    - TMmodel: To represent a trained topic model + edition functions
    - MalletTrainer: To train a topic model from a given corpus
"""

import argparse
import configparser
from pathlib import Path



class MalletTrainer(object):
    
    def __init__(self, corpusFile, outputFolder, mallet_path, 
        numTopics=None, alpha=None, optimizeInterval=None,
        numThreads=None, numIterations=None, docTopicsThreshold=None,
        sparse_thr=None, sparse_block=0, logger=None):
        """Inicializador del objeto
        """
        self._corpusFile = Path(corpusFile)
        self._numTopics = numTopics
        self._alpha = alpha
        self._optimizeInterval = optimizeInterval
        self._numThreads = numThreads
        self._numIterations = numIterations
        self._docTopicsThreshold = docTopicsThreshold
        self._outputFolder = Path(outputFolder)
        self._sparse_thr = sparse_thr
        self._sparse_block = sparse_block
        self._mallet_path = Path(mallet_path)
        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('MalletTrainer')

    def adj_settings(self):
        """Ajuste de parámetros manual"""
        self._numTopics = var_num_keyboard('int', self._numTopics,
                                    'Número de tópicos para el modelo')
        self._alpha = var_num_keyboard('float',1,
                                    'Valor para el parametro alpha')
        self._optimizeInterval = var_num_keyboard('int',self._optimizeInterval,
                                'Optimization of hyperparameters every optimize_interval iterations')
        self._numIterations = var_num_keyboard('int',self._numIterations,
                                'Iteraciones máximas para el muestreo de Gibbs')
        self._sparse_thr = var_num_keyboard('float',self._sparse_thr,
                                'Probabilidad para poda para "sparsification" del modelo')

    def fit(self):
        """Rutina de entrenamiento
        """
        config_file = self._outputFolder.joinpath('train.config')
        with config_file.open('w', encoding='utf8') as fout:
            fout.write('input = ' + self._corpusFile.as_posix() + '\n')
            fout.write('num-topics = ' + str(self._numTopics) + '\n')
            fout.write('alpha = ' + str(self._alpha) + '\n')
            fout.write('optimize-interval = ' + str(self._optimizeInterval) + '\n')
            fout.write('num-threads = ' + str(self._numThreads) + '\n')
            fout.write('num-iterations = ' + str(self._numIterations) + '\n')
            fout.write('doc-topics-threshold = ' + str(self._docTopicsThreshold) + '\n')
            #fout.write('output-state = ' + os.path.join(self._outputFolder, 'topic-state.gz') + '\n')
            fout.write('output-doc-topics = ' + \
                self._outputFolder.joinpath('doc-topics.txt').as_posix() + '\n')
            fout.write('word-topic-counts-file = ' + \
                self._outputFolder.joinpath('word-topic-counts.txt').as_posix() + '\n')
            fout.write('diagnostics-file = ' + \
                self._outputFolder.joinpath('diagnostics.xml ').as_posix() + '\n')
            fout.write('xml-topic-report = ' + \
                self._outputFolder.joinpath('topic-report.xml').as_posix() + '\n')
            fout.write('output-topic-keys = ' + \
                self._outputFolder.joinpath('topickeys.txt').as_posix() + '\n')
            fout.write('inferencer-filename = ' + \
                self._outputFolder.joinpath('inferencer.mallet').as_posix() + '\n')
            #fout.write('output-model = ' + \
            #    self._outputFolder.joinpath('modelo.bin').as_posix() + '\n')
            #fout.write('topic-word-weights-file = ' + \
            #    self._outputFolder.joinpath('topic-word-weights.txt').as_posix() + '\n')

        cmd = str(self._mallet_path) + ' train-topics --config ' + str(config_file)

        try:
            self.logger.info(f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Model training failed. Revise command')
            return

        thetas_file = self._outputFolder.joinpath('doc-topics.txt')
        #Modified to allow for non integer identifier
        cols = [k for k in np.arange(2,self._numTopics+2)]

        if self._sparse_block==0:
            self.logger.debug('-- -- Sparsifying doc-topics matrix')
            thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32, usecols=cols)
            #thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32)[:,2:]
            #Save figure to check thresholding is correct
            allvalues = np.sort(thetas32.flatten())
            step = int(np.round(len(allvalues)/1000))
            plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
            plt.semilogx([self._sparse_thr, self._sparse_thr], [0,100], 'r')
            plot_file = self._outputFolder.joinpath('thetas_dist.pdf')
            plt.savefig(plot_file)
            plt.close()
            #sparsify thetas
            thetas32[thetas32<self._sparse_thr] = 0
            thetas32 = normalize(thetas32,axis=1,norm='l1')
            thetas32_sparse = sparse.csr_matrix(thetas32, copy=True)

        else:
            self.logger.debug('-- -- Sparsifying doc-topics matrix using blocks')
            #Leemos la matriz en bloques
            ndocs = file_len(thetas_file)
            thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32,
                                     usecols=cols, max_rows=self._sparse_block)
            #Save figure to check thresholding is correct
            #In this case, the figure will be calculated over just one block of thetas
            allvalues = np.sort(thetas32.flatten())
            step = int(np.round(len(allvalues)/1000))
            plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
            plt.semilogx([self._sparse_thr, self._sparse_thr], [0,100], 'r')
            plot_file = self._outputFolder.joinpath('thetas_dist.pdf')
            plt.savefig(plot_file)
            plt.close()
            #sparsify thetas
            thetas32[thetas32<self._sparse_thr] = 0
            thetas32 = normalize(thetas32,axis=1,norm='l1')
            thetas32_sparse = sparse.csr_matrix(thetas32, copy=True)
            for init_pos in np.arange(0,ndocs,self._sparse_block)[1:]:
                thetas32_b = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32,
                                         usecols=cols, max_rows=self._sparse_block,
                                         skiprows=init_pos)
                #sparsify thetas
                thetas32_b[thetas32_b<self._sparse_thr] = 0
                thetas32_b = normalize(thetas32_b,axis=1,norm='l1')
                thetas32_b_sparse = sparse.csr_matrix(thetas32_b, copy=True)
                thetas32_sparse = sparse.vstack([thetas32_sparse, thetas32_b_sparse])

        #Recalculamos alphas para evitar errores de redondeo por la sparsification
        alphas = np.asarray(np.mean(thetas32_sparse,axis=0)).ravel()

        #Create vocabulary files
        wtcFile = self._outputFolder.joinpath('word-topic-counts.txt')
        vocab_size = file_len(wtcFile)
        betas = np.zeros((self._numTopics,vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))

        with wtcFile.open('r', encoding='utf8') as fin:
            for i,line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc,i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas,axis=1,norm='l1')
        #save vocabulary and frequencies
        with self._outputFolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            [fout.write(el+'\n') for el in vocab]
        with self._outputFolder.joinpath('vocab_freq.txt').open('w', encoding='utf8') as fout:
            [fout.write(el[0]+'\t'+str(int(el[1]))+'\n') for el in zip(vocab,term_freq)]
        self.logger.debug('-- -- Mallet training: Vocabulary files generated')

        tmodel = TMmodel(betas=betas,thetas=thetas32_sparse,alphas=alphas,
                            vocabfreq_file=self._outputFolder.joinpath('vocab_freq.txt'),
                            logger=self.logger)
        tmodel.save_npz(self._outputFolder.joinpath('modelo.npz'))

        #Remove doc-topics file. It is no longer needed
        thetas_file.unlink()

        return

if __name__ == "__main__":

    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train a Topic Model according to config file")
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")
    args = parser.parse_args()
    
    #If the training flag is activated, we need to check availability of
    #configuration file, and run the training using class MalletTrainer
    if args.train:
        configFile = Path(args.config)
        if configparser.is_file():
            cf = configparser.ConfigParser()
            cf.read(configFile)
            if cf['Training']['trainer'] == 'mallet':
                #Create Topic model
                MallTr = MalletTrainer()



        else:
            print('You need to provide a valid configuration file')


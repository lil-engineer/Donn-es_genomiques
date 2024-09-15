import math
import numpy as np

##### loadRawGenoFile(filename,ext=False)
#
#   Load and prepare a .geno file and extract some key characteristics
#
#   parameters:
#       filename : path/filename
#       ext : if set to "False", the ".geno" extension is added
#
#   returns:
#       geno_file : the raw file encoded
#       nind : the number of lines/individuals in the file
#       nsnp : the number of columns/snps per individual
#       rlen : the record length of each row
#
def loadRawGenoFile(filename,ext=True):
    if(ext):
        geno_file=open(filename, 'rb')
    else:
        geno_file=open(filename+".geno", 'rb')
        
    header=geno_file.read(20)
    nind,nsnp=[int(x) for x in header.split()[1:3]]
    rlen=max(48,int(np.ceil(nind*2/8)))
    geno_file.seek(rlen)
    return geno_file,nind,nsnp,rlen


##### unpackfullgenofile(filename)
#
#   Unpack a .geno file
#
#   parameters:
#       filename : path/filename
#
#   returns:
#       geno : the geno file as a numpy array
#       nind : the number of lines/individuals in the file
#       nsnp : the number of columns/snps per individual
#       rlen : the record length of each row
#
def unpackfullgenofile(filename):                     
    geno_file, nind,nsnp,rlen=loadRawGenoFile(filename)
    geno=np.fromfile(filename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    geno=np.unpackbits(geno,axis=1)[:,:(2*nind)]
    return geno,nind,nsnp,rlen
    
    
##### unpackAndFilterSNPs(genofilename,snpIndexes)
#
#   Unpack a geno data and pre-filter the SNPs
#
#   parameters:
#       geno : raw numpy geno file
#       snpIndexes : an index list of SNPs to keep /!\ must be extracted from a compatible SNP file with the same indexing !
#       nind : number of individuals
#
#   returns:
#       geno : the geno file as a numpy array
#
def unpackAndFilterSNPs(geno,snpIndexes,nind):
    geno=np.unpackbits(geno,axis=1)[snpIndexes,:(2*nind)]
    geno=2*geno[:,::2]+geno[:,1::2]
    return geno    
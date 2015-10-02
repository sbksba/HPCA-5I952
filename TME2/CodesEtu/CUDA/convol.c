/*
 * Programmation GPU 
 * Université Pierre et Marie Curie
 * Calcul de convolution sur une image.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>   /* pour le rint */
#include <string.h> /* pour le memcpy */
#include <time.h>   /* chronometrage */
#include <libgen.h> /* pour basename */ 
#include <sys/stat.h> /* pour mkdir */ 
#include <unistd.h>   /* pour getlogin */

#include "rasterfile.h"

#define MAX(a,b) ((a>b) ? a : b)

/** 
 * \struct Raster
 * Structure décrivant une image au format Sun Raster
 */

typedef struct {
  struct rasterfile file;  ///< Entête image Sun Raster
  unsigned char rouge[256],vert[256],bleu[256];  ///< Palette de couleur
  unsigned char *data;    ///< Pointeur vers l'image
} Raster;




double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}




/**
 * Cette procedure convertit un entier LINUX en un entier SUN 
 *
 * \param i pointeur vers l'entier à convertir
 */

void swap(int *i) {
  unsigned char s[4],*n;
  memcpy(s,i,4);
  n=(unsigned char *)i;
  n[0]=s[3];
  n[1]=s[2];
  n[2]=s[1];
  n[3]=s[0];
}

/**
 * \brief Lecture d'une image au format Sun RASTERFILE.
 *
 * Au retour de cette fonction, la structure r est remplie
 * avec les données liée à l'image. Le champ r.file contient
 * les informations de l'entete de l'image (dimension, codage, etc).
 * Le champ r.data est un pointeur, alloué par la fonction
 * lire_rasterfile() et qui contient l'image. Cette espace doit
 * être libéré après usage.
 *
 * \param nom nom du fichier image
 * \param r structure Raster qui contient l'image
 *  chargée en mémoire
 */

void lire_rasterfile(char *nom, Raster *r) {
  FILE *f;
  int i,h,w,w2;
    
  if( (f=fopen( nom, "r"))==NULL) {
    fprintf(stderr,"erreur a la lecture du fichier %s\n", nom);
    exit(1);
  }
  if (fread( &(r->file), sizeof(struct rasterfile), 1, f) < 1){
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  };    
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));
    
  if ((r->file.ras_depth != 8) ||  (r->file.ras_type != RT_STANDARD) ||
      (r->file.ras_maptype != RMT_EQUAL_RGB)) {
    fprintf(stderr,"palette non adaptee\n");
    exit(1);
  }
    
  /* composante de la palette */
  if (fread(&(r->rouge),r->file.ras_maplength/3,1,f) < 1){ 
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  };    
  if (fread(&(r->vert), r->file.ras_maplength/3,1,f) < 1){
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  };    
  if (fread(&(r->bleu), r->file.ras_maplength/3,1,f) < 1){
    fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  };    
    
  if ((r->data=malloc(r->file.ras_width*r->file.ras_height))==NULL){
    fprintf(stderr,"erreur allocation memoire\n");
    exit(1);
  }

  /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
  h=r->file.ras_height;
  w=r->file.ras_width;
  w2=((w + 1) & ~1); /* multiple of 2 greater or equal */
  //  printf("Dans lire_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  for (i=0; i<h; i++){
    if (fread(r->data+i*w,w,1,f) < 1){
      fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
    }
    if (w2-w > 0){ fseek(f, w2-w, SEEK_CUR); } 
  }

  fclose(f);
}

/**
 * Sauve une image au format Sun Rasterfile
 */

void sauve_rasterfile(char *nom, Raster *r)     {
  FILE *f;
  int i,h,w,w2;
  
  if( (f=fopen( nom, "w"))==NULL) {
    fprintf(stderr,"erreur a l'ecriture du fichier %s\n", nom);
    exit(1);
  }
    
  h=r->file.ras_height;
  w=r->file.ras_width;

  /* en-tete : */
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));   
  fwrite(&(r->file),sizeof(struct rasterfile),1,f);

  /* composante de la palette : */
  fwrite(&(r->rouge),256,1,f);
  fwrite(&(r->vert),256,1,f);
  fwrite(&(r->bleu),256,1,f);

  /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
  w2=((w + 1) & ~1); /* multiple of 2 greater or equal */
  //  printf("Dans lire_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  for (i=0; i<h; i++){
    if (fwrite(r->data+i*w,w,1,f) < 1){
      fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__); 
    }
    if (w2-w > 0){ /* padding */
      unsigned char zeros[1]={0}; 
      if (w2-w != 1){ fprintf(stderr, "Error in sauve_rasterfile(): w2-w != 1 \n"); }
      if (fwrite(zeros, w2-w, 1, f) < 1){
	fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__); 
      }
    } 
  }

  /* re-order bytes in original order so that 
     sauve_rasterfile() can be called multiple 
     times with the same 'r' variable */
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));

  fclose(f);
}




/**
 * Conversion d'une image avec un "unsigned char" par pixel en une image 
 * avec un "float" par pixel. 
 */

void convert_uchar2float_image(unsigned char*p_ua, float *p_f, int h, int w){
  int i,j;
  
  for (i=0; i<h; i++){
    for(j=0; j<w; j++){
      p_f[i*w+j] = (float) p_ua[i*w+j]; 
    }
  }
}



/**
 * Conversion d'une image avec un "float" par pixel en une image 
 * avec un "unsigned char" par pixel. 
 */

void convert_float2uchar_image(float *p_f, unsigned char*p_ua, int h, int w){
  int i,j;
  
  for (i=0; i<h; i++){
    for(j=0; j<w; j++){
      p_ua[i*w+j] = (unsigned char) rintf(p_f[i*w+j]); 
    }
  }
}




/**
 * Convolution d'une image par un filtre prédéfini
 * \param choix choix du filtre (voir la fonction filtre())
 * \param tab pointeur vers l'image
 * \param nbl, nbc dimension de l'image
 *
 * \sa filtre()
 */

int convolution(float buf_src[], float buf_tgt[], int nbl,int nbc) {
  float numerateur, denominateur;
  float poids_central; 
  int i,j;
  
  for(i=0 ; i<nbl ; i++){
    for(j=0 ; j<nbc ; j++){
      
      /*** filtre moyenneur CONVOL_MOYENNE2 (filtre moyenneur avec 
       * un poid central plus fort):
       * Rq: pour les bords, moyenne avec uniquement les cases presentes */
      denominateur = 0.0f; 
      numerateur = 0.0f;
      if (i<nbl-1){
	numerateur += buf_src[(i+1)*nbc+j]; ++denominateur;
	if (j>0){     numerateur += buf_src[(i+1)*nbc+j-1]; ++denominateur; }
	if (j<nbc-1){ numerateur += buf_src[(i+1)*nbc+j+1]; ++denominateur; }
      }
      if (j>0){     numerateur += buf_src[(i)*nbc+j-1]; ++denominateur; }
      if (j<nbc-1){ numerateur += buf_src[(i)*nbc+j+1]; ++denominateur; } 
      if (i>0){
	numerateur +=  buf_src[(i-1)*nbc+j]; ++denominateur; 
	if (j>0){     numerateur += buf_src[(i-1)*nbc+j-1]; ++denominateur; }
	if (j<nbc-1){ numerateur += buf_src[(i-1)*nbc+j+1]; ++denominateur; }
      }
      poids_central = denominateur*0.5f; /* poids centrale = 50% autres poids */
      numerateur   += poids_central*buf_src[(i)*nbc+j];
      denominateur += poids_central;

      buf_tgt[i*nbc+j] = numerateur/denominateur;	
   } /* for j */
  } /* for i */
}


/**
 * Effectue 'nbiter' convolutions et retourne
 * le pointeur vers le buffer contenant la derniere convolution. 
 */

float *multiples_convolutions(float buf[], float buf_aux[], int nbl,int nbc, int nbiter) {
  int n,i;
  float* bufs[2] = { buf, buf_aux }; 

  for(n=0 ; n<nbiter ; n++){
    i = n%2; 
    convolution(bufs[i], bufs[1-i], nbl, nbc);
  } /* for n */
  
  return bufs[n%2];
}


/**
 * Definition dans gpu_convol.cu 
 */

extern float *gpu_multiples_convolutions(float buf[], float buf_aux[], int nbl,int nbc, int nbiter, int nbThreadsParBloc
#ifdef DOWNLOAD
				  , Raster r, char *nom_sortie
#endif 				       				       				       
);


/**
 * Interface utilisateur
 */

static char usage [] = "Usage : %s <nom image SunRaster> <nbiter> <nbThreadsParBloc> \n";

/*
 * Partie principale
 */

int main(int argc, char *argv[]) {

  /* Variables se rapportant a l'image elle-meme */
  Raster r;
  int    w, h;	/* nombre de lignes et de colonnes de l'image */
  char nom_sortie[100] = "";
  char *nom_base = basename(argv[1]);
  char nom_rep[30] = "";

  /* Variables liees au traitement de l'image */
  int 	 nbiter;		/* nombre d'iterations */

  /* Variables liees au chronometrage */
  double debut, fin;

  /* Buffers: */
  float *buf, *buf_aux, *buf_res = NULL;
  float *cpu_buf, *cpu_buf_aux, *cpu_buf_res = NULL; 

  /* Pour GPU : */
  int nbThreadsParBloc;

  if (argc < 4) {
    fprintf( stderr, usage, argv[0]);
    return 1;
  }
      
  /* Saisie des parametres */
  nbiter = atoi(argv[2]);
  nbThreadsParBloc = atoi(argv[3]);
        
  /* Lecture du fichier Raster */
  lire_rasterfile( argv[1], &r);
  h = r.file.ras_height;
  w = r.file.ras_width;
  printf("Image %s : %dx%d\n", argv[1], h, w);

  /* Preparation sauvegarde finale : */
  sprintf(nom_rep, "/tmp/%s", getlogin());
  //printf("nom_rep = %s\n", nom_rep); 
  mkdir(nom_rep, S_IRWXU);
  sprintf(nom_sortie, "%s/post-convolution_%s", nom_rep, nom_base);
  //printf("nom_sortie = %s\n", nom_sortie); 

  /* Allocation memoire : */
  buf     = (float *) malloc(h * w * sizeof(float));
  buf_aux = (float *) malloc(h * w * sizeof(float));
  cpu_buf     = (float *) malloc(h * w * sizeof(float));
  cpu_buf_aux = (float *) malloc(h * w * sizeof(float));

  /* Conversion : unsigned char -> float */
  convert_uchar2float_image(r.data, buf, h, w);
  convert_uchar2float_image(r.data, cpu_buf, h, w);    
    
  /* debut du chronometrage */
  debut = my_gettimeofday();            

  /* La convolution a proprement parler */
  buf_res = gpu_multiples_convolutions(buf, buf_aux, h, w, nbiter, nbThreadsParBloc
#ifdef DOWNLOAD
				       ,r, nom_sortie
#endif 				       				       				       
				       );

  /* fin du chronometrage */
  fin = my_gettimeofday();
  printf("Temps total : %g seconde(s) \tNb convolutions/s : %g\n",
	 fin - debut, nbiter/(fin - debut));
    
  /* comparaison avec execution CPU : */
  {
    int i,j,same=1; 

    printf("Verification des resultats avec le calcul CPU : "); fflush(stdout);  

    debut = my_gettimeofday();            
    cpu_buf_res = multiples_convolutions(cpu_buf, cpu_buf_aux, h, w, nbiter);
    fin = my_gettimeofday();

    for (i=0; i<h; i++){
      for(j=0; j<w; j++){
	if (fabsf(cpu_buf_res[i*w+j] - buf_res[i*w+j]) > 0.01f){
	  fprintf(stderr, "\n cpu_buf_res[%d,%d]=%g est different de buf_res[%d,%d]=%g ", 
		 i, j, cpu_buf_res[i*w+j], i, j, buf_res[i*w+j]); 
	  same=0; 
	} 
      }
    }
    printf((same == 1 ?  " ok !\n" : "\n -> pas ok ! \n" ));

    printf("Temps total CPU : %g seconde(s) \tNb convolutions/s : %g\n",
	 fin - debut, nbiter/(fin - debut));
  }
  
  /* Conversion : float -> unsigned char */
  convert_float2uchar_image(buf_res, r.data, h, w);

  /* Sauvegarde du fichier Raster */
  { 
    char nom_sortie_nbIter[100]; 
    sprintf(nom_sortie_nbIter, "%s_nbIter%d.ras", nom_sortie, nbiter);
    sauve_rasterfile(nom_sortie_nbIter, &r);
  }

  /* Liberation memoire : */
  free(buf);
  free(buf_aux); 
  free(cpu_buf); 
  free(cpu_buf_aux);

  return 0;
}


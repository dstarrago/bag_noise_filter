/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.bag;

import SIFilters.KeelNoiseFilter;
import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.experiment.Stats;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.SimpleKMeans;

/**
 *
 * @author Danel
 */
public class CCEMappingBagNoiseFilter extends AbstractBagNoiseFilter {
  
  public final int maxNumClusters = 5;
  
  private AbstractClusterer clusterer;

  public CCEMappingBagNoiseFilter(Instances dataset, Class<? extends KeelNoiseFilter> siFilterClass) {
    super(dataset, siFilterClass);
  }

  @Override
  protected Instances mapsToSingleInstance(Instances miData) throws Exception {
    /**
     * Create an empty dataset with the appropriate single-instance structure.
     */
    
    // N * (N - 1) / 2 + (N - 1)
    
    ArrayList<Attribute> attrInfo = new ArrayList();
    for (int g = 0; g < maxNumClusters; g++) {
      for (int at = 0; at < miData.attribute(1).relation().numAttributes(); at++) {
        Attribute attr = miData.attribute(1).relation().attribute(at).copy(String.format("p%d_%s", (g+1), 
                miData.attribute(1).relation().attribute(at).name()));
        attrInfo.add(attr);
      }
    }
    attrInfo.add(miData.attribute(2).copy("Class"));
    Instances mappedData = new Instances("si" + miData.relationName(), attrInfo, miData.numInstances());
    mappedData.setClassIndex(mappedData.numAttributes() - 1);

    clusterer = new SimpleKMeans();
    ((SimpleKMeans)clusterer).setNumClusters(maxNumClusters);
    
//    clusterer = new Cobweb();

    Instances allInstances = new Instances(miData.attribute(1).relation(), 0);
    for (int i = 0; i < miData.numInstances(); i++) {
      allInstances.addAll(miData.attribute(1).relation(i));
    }
    clusterer.buildClusterer(allInstances);
    
    for (int i = 0; i < miData.numInstances(); i++) {
      Instances bag = miData.attribute(1).relation(i);
      
      ArrayList<Instance>[] inBagCluster = new ArrayList[maxNumClusters];
      for (int j = 0; j < maxNumClusters; j++) {
        inBagCluster[j] = new ArrayList<>();
      }

      for (int j = 0; j < bag.numInstances(); j++) {
        Instance x = bag.get(j);
        int c = clusterer.clusterInstance(x);
        inBagCluster[c].add(x);
      }
      
      double[] patternAttrVals = new double[maxNumClusters * bag.numAttributes() + 1];
      for (int g = 0; g < maxNumClusters; g++) {
        for (int j = 0; j < bag.numAttributes(); j++) {
          Stats stats = new Stats();
          for (int k = 0; k < inBagCluster[g].size(); k++) {
            stats.add(inBagCluster[g].get(k).value(j));
          }
          stats.calculateDerived();
          patternAttrVals[g * bag.numAttributes() + j] = stats.mean;
        }
      }
      
      patternAttrVals[patternAttrVals.length - 1] = miData.get(i).classValue();
      mappedData.add(new DenseInstance(1, patternAttrVals));
    }
    
    return mappedData;
  }
  
}

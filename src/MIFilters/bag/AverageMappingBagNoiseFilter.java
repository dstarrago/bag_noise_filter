/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package MIFilters.bag;

import SIFilters.KeelNoiseFilter;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.Filter;
import weka.core.DenseInstance;

/**
 *
 * @author Danel
 */
public class AverageMappingBagNoiseFilter extends AbstractBagNoiseFilter {

  public AverageMappingBagNoiseFilter(Instances dataset, Class<? extends KeelNoiseFilter> siFilterClass) {
    super(dataset, siFilterClass);
  }

  @Override
  protected Instances mapsToSingleInstance(Instances miData) throws Exception {
    /**
     * Create an empty dataset with the appropriate single-instance structure:
     * same attributes as the MIL data plus the class attribute.
     */
    Instances siData = new Instances(miData.attribute(1).relation(), miData.numInstances());
    Add fAdd = new Add();
    try {
      fAdd.setOptions(new String[]{"-T", "NOM", "-N", "class", "-L", "0,1"});
    } catch (Exception ex) {
      Logger.getLogger(AverageMappingBagNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      throw new Exception("Hubo problemas al agregar el atributo clase en el método de mapeo");
    }
    fAdd.setInputFormat(siData);
    siData = Filter.useFilter(siData, fAdd);
    
    for (int i = 0; i < miData.numInstances(); i++) {
      Instances bag = miData.attribute(1).relation(i);
      double[] attrVals = new double[siData.numAttributes()];
      for (int j = 0; j < bag.numAttributes(); j++) {
        attrVals[j] = bag.meanOrMode(j);
      }
      attrVals[bag.numAttributes()] = miData.get(i).classValue();
      siData.add(new DenseInstance(1, attrVals));
    }
    return siData;
  }
  
}

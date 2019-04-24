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
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.Filter;
import weka.core.DenseInstance;
import weka.experiment.Stats;

/**
 *
 * @author Danel
 */
public class MinMaxMappingBagNoiseFilter extends AbstractBagNoiseFilter {

  public MinMaxMappingBagNoiseFilter(Instances dataset, Class<? extends KeelNoiseFilter> siFilterClass) {
    super(dataset, siFilterClass);
  }

  @Override
  protected Instances mapsToSingleInstance(Instances miData) throws Exception {
    /**
     * Create an empty dataset with the appropriate single-instance structure:
     * same attributes as the MIL data for min values, another copy of the same 
     * attributes of the MIL data for max values and the class attribute.
     */
    Instances siMinData = new Instances(miData.attribute(1).relation(), miData.numInstances());
    Instances siMaxData = new Instances(miData.attribute(1).relation(), miData.numInstances());

    for (int i = 0; i < miData.numInstances(); i++) {
      Instances bag = miData.attribute(1).relation(i);
      double[] attrMinVals = new double[bag.numAttributes()];
      double[] attrMaxVals = new double[bag.numAttributes()];
      for (int j = 0; j < bag.numAttributes(); j++) {
        Stats stats = new Stats();
        for (int k = 0; k < bag.numInstances(); k++) {
          stats.add(bag.get(k).value(j));
        }
        attrMinVals[j] = stats.min;
        attrMaxVals[j] = stats.max;
      }
      siMinData.add(new DenseInstance(1, attrMinVals));
      siMaxData.add(new DenseInstance(1, attrMaxVals));
    }
    RenameAttribute raFilter = new RenameAttribute();
    try {
      raFilter.setOptions(new String[]{"-find", "(^)", "-replace", "min_", "-R", "first-last"});
      raFilter.setInputFormat(siMinData);
      siMinData = Filter.useFilter(siMinData, raFilter);
      raFilter.setOptions(new String[]{"-find", "(^)", "-replace", "max_", "-R", "first-last"});
      raFilter.setInputFormat(siMaxData);
      siMaxData = Filter.useFilter(siMaxData, raFilter);
    } catch (Exception ex) {
      Logger.getLogger(MinMaxMappingBagNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      throw new Exception("There was some problem renaming attributes to mix-max mapping method");
    }
    Instances siData = Instances.mergeInstances(siMinData, siMaxData);
    Add filter = new Add();
    try {
      filter.setOptions(new String[]{"-T", "NOM", "-N", "class", "-L", "0,1"});
    } catch (Exception ex) {
      Logger.getLogger(MinMaxMappingBagNoiseFilter.class.getName()).log(Level.SEVERE, null, ex);
      throw new Exception("There was some problem when adding the class attribute in the mapped dataset");
    }
    filter.setInputFormat(siData);
    siData = Filter.useFilter(siData, filter);
    siData.setClassIndex(siData.numAttributes() - 1);
    for (int i = 0; i < miData.numInstances(); i++) {
      siData.get(i).setClassValue(miData.get(i).classValue());
    }
    return siData;
  }
  
}

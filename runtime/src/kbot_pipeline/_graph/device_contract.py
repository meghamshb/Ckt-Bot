"""One physical terminal inventory and required-role contract for assembly."""

TWO_PIN=frozenset(('resistor','capacitor','inductor','diode','switch','voltage_source',
    'current_source','dependent_voltage_source','dependent_current_source','impedance_block'))
REQUIRED_ROLES={'voltage_source':{'positive','negative'},'dependent_voltage_source':{'positive','negative'},
    'current_source':{'from','to'},'dependent_current_source':{'from','to'},'diode':{'anode','cathode'},
    'op_amp':{'noninverting','inverting','output'},'transistor_bjt':{'base','collector','emitter'}}


def physical_arities(kind):
    if kind in TWO_PIN:return (2,)
    return {'ground':(1,),'op_amp':(3,5),'transistor_bjt':(3,),
        'two_port_block':(4,),'ideal_transformer':(4,)}.get(kind,())


def required_roles(kind,count):
    result=set(REQUIRED_ROLES.get(kind,()))
    if kind=='op_amp' and count==5:result.update(('supply_positive','supply_negative'))
    return result


def normalize_component_models(components,models):
    """Describe observed physical inventory without inventing pins or supply roles."""
    kinds={c['id']:c['type'] for c in components}
    for model in models:
        kind=kinds.get(model['component_id']);count=model.get('observed_contacts')
        model['allowed_physical_arities']=list(physical_arities(kind))
        model['required_semantic_roles']=sorted(required_roles(kind,count))
        if kind=='op_amp':
            model['terminal_model']='op_amp_signal' if count==3 else 'op_amp_with_supplies' if count==5 else 'op_amp_inventory_unresolved'
            model['expected_arity']=count if count in (3,5) else None
            model['analysis_model_supported']=count==3
            model['analysis_scope']='ideal_linear_three_signal_pins' if count==3 else 'five_pin_analysis_not_implemented' if count==5 else 'inventory_unresolved'

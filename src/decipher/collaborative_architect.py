import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# --- Funções de IA Especializadas e Utilitárias ---

def _protected_exp(x):
    with np.errstate(over='ignore'):
        return np.exp(x)
exp_func = make_function(function=_protected_exp, name='exp', arity=1)

def find_formula(n, sequence, brain, feature_names, generations=15, population_size=2000):
    """Função genérica para um especialista de IA encontrar uma fórmula."""
    est = SymbolicRegressor(population_size=population_size, generations=generations,
                            stopping_criteria=1e-5, verbose=1,
                            feature_names=feature_names, function_set=brain,
                            const_range=(-1., 1.), random_state=42)
    est.fit(n, sequence)
    return est

# --- O Arquiteto Colaborativo ---

def run_collaborative_architect(sequence, num_iterations=3):
    """
    O 'Arquiteto' que gere a colaboração iterativa entre as IAs especialistas.
    """
    print("\n" + "="*60)
    print("🏛️ INICIANDO ARQUITETURA 'CÉREBRO COLETIVO' DO CYGNUS 🏛️")
    print("="*60)
    
    n = np.arange(len(sequence)).reshape(-1, 1)
    
    # Estimativas iniciais
    trend_prediction = np.ones_like(sequence)
    oscillation_prediction = np.ones_like(sequence)
    
    # O CÉREBRO COLETIVO: O LOOP DE REFINAMENTO ITERATIVO
    for i in range(num_iterations):
        print(f"\n--- CICLO DE COLABORAÇÃO: ITERAÇÃO {i+1}/{num_iterations} ---")

        # 1. TrendHunter refina a sua hipótese
        de_oscillated_signal = sequence / oscillation_prediction
        de_oscillated_signal = np.nan_to_num(de_oscillated_signal, nan=1.0, posinf=1.0, neginf=1.0)
        
        trend_brain = ('add', 'sub', 'mul', 'div', 'log', exp_func)
        print("\n--- 🧠 TrendHunter está a refinar a tendência... ---")
        trend_model = find_formula(n, de_oscillated_signal, trend_brain, ['n'])
        trend_prediction = trend_model.predict(n)
        trend_prediction = np.nan_to_num(trend_prediction, nan=1.0, posinf=1.0, neginf=1.0)
        trend_prediction[np.abs(trend_prediction) < 1e-6] = 1e-6
        
        # 2. RhythmFinder refina a sua hipótese
        detrended_signal = sequence / trend_prediction
        detrended_signal = np.nan_to_num(detrended_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        rhythm_brain = ('add', 'sub', 'mul', 'sin', 'cos')
        print("\n--- 🎶 RhythmFinder está a refinar a oscilação... ---")
        oscillation_model = find_formula(n, detrended_signal, rhythm_brain, ['n'])
        oscillation_prediction = oscillation_model.predict(n)
        oscillation_prediction = np.nan_to_num(oscillation_prediction, nan=1.0, posinf=1.0, neginf=1.0)

    # 3. Síntese Final do Arquiteto
    print("\n" + "="*60)
    print("🏆 SÍNTESE FINAL DO CÉREBRO COLETIVO 🏆")
    print("="*60)
    print(f"📈 Hipótese Final de Tendência (T): {trend_model._program}")
    print(f"🌊 Hipótese Final de Oscilação (O): {oscillation_model._program}")
    
    final_formula_str = f"mul({trend_model._program}, {oscillation_model._program})"
    print(f"\n🧩 Hipótese da Fórmula Combinada (T * O): {final_formula_str}")
    
    # 4. Verificação Final
    final_prediction = trend_model.predict(n) * oscillation_model.predict(n)
    final_error = np.mean(np.abs(sequence - final_prediction))
    print(f"\n📊 Erro (Fitness) da Fórmula Combinada Final: {final_error:.6f}")

    if final_error < 0.01:
        print("\n✅ Veredito: SUCESSO! A equipe de IAs convergiu e desvendou a regra oculta.")
    else:
        print("\n❌ Veredito: FALHA. A equipe não conseguiu convergir para uma solução precisa.")

if __name__ == "__main__":
    # O teste de estresse que a arquitetura anterior falhou
    print("--- Gerando sinal de teste com a regra S(n) = exp(-0.1n) * sin(0.5n) ---")
    test_rule = lambda n: np.exp(-0.1 * n) * np.sin(0.5 * n)
    test_sequence = np.array([test_rule(i) for i in range(100)])
    
    run_collaborative_architect(test_sequence, num_iterations=3)